"""Expected value with a ratio-aware penalty, then policy gates that can veto it.

Two ideas here. The first is that the cost of losing a fight is not a constant:
a merchant at 0.1% chargeback ratio can afford to lose one, a merchant at 0.9%
is one bad month from a monitoring programme, and the same lost dispute costs
those two merchants completely different amounts. The EV maths only works if the
penalty term knows where the merchant is standing.

The second is that expected value is necessary and not sufficient. EV will
happily refund the same customer eleven times because each refund is individually
cheap. The gates exist to stop locally rational decisions adding up to a policy
nobody would sign off on, and they run after EV and can overrule it.

decide() never sees would_win_if_fought. run_batch() resolves outcomes with it
only after the decision is made, which is the simulation standing in for the
weeks a real representment takes to come back.
"""

import os
import sys
from pathlib import Path

import pandas as pd

import actions
import audit
import evaluate
import llm
from evaluate import FEE_SAVED, REPRESENT_COST

# ponytail: every constant below is illustrative. Real numbers come from the
# acquirer's fee schedule and the network's published programme thresholds, and
# they differ by region, MCC and processing volume. The shape is the claim here;
# the magnitudes are a plausible stand-in.
MONITORING_THRESHOLD = 0.01     # ~1% is where Visa/Mastercard programmes begin
PENALTY_FLOOR = 500             # a lost fight at a healthy ratio: just the admin
PENALTY_CEILING = 50_000        # at the threshold: fines, remediation, review
PENALTY_EXPONENT = 3            # convexity -- see ratio_penalty

LIFETIME_DEFLECTION_BUDGET = 2  # refunds-on-demand a customer gets, ever
VELOCITY_LIMIT = 2              # ...and at most this many in any
VELOCITY_WINDOW_DAYS = 90       # ...rolling window this long
ESCALATION_STEP = 500           # each prior deflection demands this much more EV margin
LOW_CONFIDENCE = (0.40, 0.60)   # neither side of the coin is worth acting on alone
HUMAN_REVIEW_COST = 250         # analyst time to read one queued alert and call it

EPS = 1e-6                      # p_win saturates at exactly 0.0 and 1.0 (isotonic)


def ratio_penalty(current_ratio: float, threshold: float = MONITORING_THRESHOLD) -> float:
    """Rupee cost of one more lost fight, given where the ratio already sits.

    Cubic in the fraction of the threshold consumed:

        penalty = FLOOR + (CEILING - FLOOR) * (ratio / threshold) ** 3

    Convex on purpose. At 20% of the threshold it returns 896 -- losing a fight
    is nearly free, so fight anything with a decent case. At 85% it returns
    30,893, which is larger than most of the disputes in this dataset and flips
    the EV on all but the strongest evidence. The merchant stops fighting long
    before the ratio touches the line, which is the entire point: the penalty has
    to bite while there is still room to correct, not after the fines arrive.

    Clipped at the threshold. Past that the merchant is already in remediation
    and the marginal cost of one more chargeback stops being the interesting
    question -- modelling that properly needs a real programme fee schedule.
    """
    assert current_ratio >= 0, f"negative ratio: {current_ratio}"
    consumed = min(current_ratio / threshold, 1.0)
    return PENALTY_FLOOR + (PENALTY_CEILING - PENALTY_FLOOR) * consumed ** PENALTY_EXPONENT


def new_state(chargebacks: int, transactions: int) -> dict:
    """Mutable merchant state. `deflections` maps customer -> days they were
    granted a refund, which is what both deflection gates read."""
    assert transactions > 0, "ratio needs a denominator"
    return {"chargebacks": chargebacks, "transactions": transactions, "deflections": {}}


def current_ratio(state: dict) -> float:
    return state["chargebacks"] / state["transactions"]


def decide(alert: dict, p_win: float, state: dict, day: int, kill_switch: bool = False,
           llm: dict | None = None) -> dict:
    """One alert in, one auditable record out. No side effects on state.

    p_win arrives as an argument rather than being fetched here so this stays
    testable on hand-built alerts, and so the deterministic money logic is
    readable without a model in the loop.

    Deliberately does not write to the audit log: this is the pure function, and
    persistence belongs to the caller. run_batch is the money path and logs every
    record it produces. Anything else that acts on a returned record and skips
    audit.log has broken the rule, and no assert here can catch that.

    `llm` is the optional output of llm.extract_features. Only
    has_internal_contradiction is read, and only by the escalating_margin gate:
    it raises the required EV margin by one ESCALATION_STEP, so it can flip a
    deflection -- including a first one -- whose margin is under ₹500. Measured
    ceiling: forcing the flag on for every alert in the 600-row test batch
    changes 4 decisions and moves net by ~0.8%. It never enters ev_fight,
    ev_refund or p_win. Passing None, or a record from a failed extraction,
    changes nothing: the neutral default is False, so an LLM outage cannot cost
    a customer their refund.
    """
    # CLAUDE.md hard rule: isotonic saturates, so p_win can be exactly 0.0 or
    # 1.0. Nothing below divides by it today, but a probability of exactly 1.0
    # states that a dispute cannot be lost, which no evidence supports.
    p = min(max(p_win, EPS), 1.0 - EPS)

    amount = float(alert["amount"])
    ratio = current_ratio(state)
    penalty = ratio_penalty(ratio)

    ev_refund = -amount + FEE_SAVED
    # The penalty sits outside the (1 - p) term: filing the representment lets
    # the chargeback formalise, and it counts against the ratio from filing --
    # winning later does not take it back off. Fighting pays the ratio price
    # win or lose; only the refund amount itself is at stake in the fight.
    ev_fight = -REPRESENT_COST - penalty - (1.0 - p) * amount
    ev_decision = "fight" if ev_fight > ev_refund else "refund"

    customer = alert["customer_id"]
    history = state["deflections"].get(customer, [])
    recent = [d for d in history if day - d < VELOCITY_WINDOW_DAYS]

    checked: list[str] = []
    passed: list[str] = []
    action, reason = ev_decision, f"EV prefers {ev_decision}"

    def gate(name: str, ok: bool) -> bool:
        checked.append(name)
        if ok:
            passed.append(name)
        return ok

    if not gate("kill_switch", not kill_switch):
        action, reason = "queue", "kill switch engaged, no automatic money movement"
    elif not gate("confidence_band", not LOW_CONFIDENCE[0] <= p <= LOW_CONFIDENCE[1]):
        action, reason = "queue", f"P(win) {p:.2f} inside the coin-flip band, needs a human"
    elif ev_decision == "refund":
        # Deflection gates only. Nothing here vetoes a decision to fight: the
        # gates exist to stop the merchant being farmed for refunds, and fighting
        # is what happens when a refund is refused.
        # A complaint that contradicts itself buys one notch less benefit of the
        # doubt: the required margin rises by one ESCALATION_STEP, exactly as if
        # the customer had already been refunded once. On a marginal case that
        # single notch can tip even a first deflection to fight -- deliberate
        # teeth, not a side effect -- but it never touches p_win, ev_fight or
        # ev_refund, and a customer whose EV case clears the step keeps their
        # refund, contradiction or not. (As with every gate, a changed action
        # changes merchant state for later alerts; that is policy working, not
        # LLM output leaking into the arithmetic.)
        contradiction = bool(llm and llm.get("has_internal_contradiction"))
        margin_required = ESCALATION_STEP * (len(history) + contradiction)
        if not gate("lifetime_budget", len(history) < LIFETIME_DEFLECTION_BUDGET):
            action, reason = "fight", (
                f"lifetime deflection budget spent ({len(history)}/{LIFETIME_DEFLECTION_BUDGET})"
            )
        elif not gate("velocity_cap", len(recent) < VELOCITY_LIMIT):
            action, reason = "fight", (
                f"{len(recent)} deflections in the last {VELOCITY_WINDOW_DAYS} days"
            )
        elif not gate("escalating_margin", (ev_refund - ev_fight) >= margin_required):
            # The reason must name what actually raised the bar. "repeat
            # deflection" on a customer with zero prior deflections is the kind
            # of audit line that falls apart the moment anyone reads it back.
            drivers = ", ".join(
                ([f"{len(history)} prior deflections"] if history else [])
                + (["contradictory complaint"] if contradiction else [])
            )
            action, reason = "fight", (
                f"EV margin {ev_refund - ev_fight:,.0f} below the "
                f"{margin_required:,.0f} required ({drivers})"
            )

    return {
        "alert_id": alert["alert_id"],
        "customer_id": customer,
        "amount": amount,
        "p_win": p,
        "ev_fight": ev_fight,
        "ev_refund": ev_refund,
        "current_ratio": ratio,
        "ratio_penalty": penalty,
        "ev_decision": ev_decision,
        "gates_checked": checked,
        "gates_passed": passed,
        "final_action": action,
        "reason": reason,
        "llm_status": (llm or {}).get("llm_status", "not_called"),
        "llm_contradiction": bool(llm and llm.get("has_internal_contradiction")),
    }


def apply_outcome(state: dict, record: dict, day: int, won: bool | None) -> None:
    """Advance merchant state after a decision resolves.

    A deflection spends budget and never becomes a chargeback -- that is the
    product. EVERY fight becomes one, won or lost: under VDMP-style accounting
    the chargeback counts against the ratio from the moment it is filed, and a
    representment won weeks later does not remove it. A queued alert moves
    nothing, because a human has not acted on it yet.
    """
    if record["final_action"] == "refund":
        state["deflections"].setdefault(record["customer_id"], []).append(day)
    elif record["final_action"] == "fight" and won is not None:
        state["chargebacks"] += 1


def run_batch(alerts: pd.DataFrame, p_win, state: dict, kill_switch: bool = False) -> pd.DataFrame:
    """Decide a whole batch in order, letting state move underneath it.

    Calls llm.extract_features per alert. With DISABLE_LLM=1, no API key, or any
    failure at all that returns the neutral defaults instantly and the batch is
    bit-for-bit identical -- which is the property worth having, not an accident.

    ponytail: the simulator emits no timestamps, so the batch is spread evenly
    across a year to give the velocity window something to measure. Real alerts
    arrive with a date; emitting one from simulator.py is the fix, and the same
    missing field is what forces the share-count leakage noted in features.py.
    """
    days = [round(i * 365 / max(len(alerts) - 1, 1)) for i in range(len(alerts))]
    records = []
    for day, (_, alert), p in zip(days, alerts.iterrows(), p_win):
        extracted = llm.extract_features(alert["complaint_text"])
        record = decide(alert.to_dict(), float(p), state, day, kill_switch, extracted)
        if record["final_action"] == "refund":
            # Execution runs its own write-ahead log lines; the decision record
            # carries the outcome so one row tells the whole story. Simulated
            # alerts have no payment id, so one is derived -- real alerts arrive
            # with the pay_... reference attached.
            execution = actions.execute_refund(
                payment_id=str(alert.get("payment_id", f"pay_sim_{record['alert_id']}")),
                amount=record["amount"], alert_id=record["alert_id"])
            record["execution_status"] = execution["execution_status"]
            record["provider_refund_id"] = execution["provider_refund_id"]
        else:
            record["execution_status"], record["provider_refund_id"] = "not_applicable", None
        # Logged before the outcome is known, which is the honest ordering: the
        # log holds what was decided and why, not what it turned out to be worth.
        audit.log(record)
        # The label is read only here, after the decision exists, standing in for
        # the weeks a real representment takes to come back.
        won = bool(alert["would_win_if_fought"]) if record["final_action"] == "fight" else None
        apply_outcome(state, record, day, won)
        records.append(record)
    return pd.DataFrame(records)


def bracket_cost(alerts: pd.DataFrame, records: pd.DataFrame, penalty=None) -> dict:
    """Net rupees for a batch, with the human queue costed two ways.

    A queued alert is not free and it is not resolved. Reporting only the 540
    alerts the system acted on would flatter it by hiding the 60 it declined to
    handle, so the queue gets bracketed instead: an analyst who calls every one
    correctly, and an analyst who refunds the lot. Neither is realistic. The true
    figure is somewhere between them, and saying so is more honest than picking a
    middle number and defending it.

    `penalty` defaults to evaluate.py's flat RATIO_PENALTY. That is the fixed
    yardstick every strategy on the leaderboard is measured against -- the
    dynamic penalty is what the decision rule *believes* a lost fight will cost
    given where the ratio sits, and scoring one strategy on its own beliefs
    while scoring the rest on a constant compares nothing. Pass the per-alert
    dynamic penalty to see the same batch under its own assumptions.
    """
    if penalty is None:
        penalty = evaluate.RATIO_PENALTY
    amount = alerts["amount"].to_numpy()
    won = alerts["would_win_if_fought"].astype(bool).to_numpy()
    action = records["final_action"].to_numpy()

    refund_payoff = -amount + FEE_SAVED
    # Penalty on every fight, won or lost: the chargeback counted from filing.
    fight_payoff = (pd.Series(float(-REPRESENT_COST), index=alerts.index) - penalty).where(
        won, -amount - REPRESENT_COST - penalty
    ).to_numpy()

    acted = (action == "refund") * refund_payoff + (action == "fight") * fight_payoff
    queued = action == "queue"
    # Best case: the analyst has the outcome in hand and picks the better branch.
    # Worst case: the analyst does what a cautious human does and refunds.
    best = acted + queued * (-HUMAN_REVIEW_COST + pd.DataFrame(
        {"r": refund_payoff, "f": fight_payoff}).max(axis=1).to_numpy())
    worst = acted + queued * (-HUMAN_REVIEW_COST + refund_payoff)

    per_1000 = lambda v: v.sum() / len(alerts) * 1000
    return {
        "n": len(alerts),
        "queued": int(queued.sum()),
        "review_cost": HUMAN_REVIEW_COST * int(queued.sum()),
        "best": per_1000(best),
        "worst": per_1000(worst),
        "acted_only": acted[~queued].sum() / max((~queued).sum(), 1) * 1000,
    }


def _alert(alert_id: str, customer_id: str, amount: float) -> dict:
    return {"alert_id": alert_id, "customer_id": customer_id, "amount": amount}


def main() -> None:
    # CLAUDE.md hard rule: self-checks never make live calls. setdefault keeps
    # an explicit DISABLE_LLM=0 able to override for a deliberate live run.
    os.environ.setdefault("DISABLE_LLM", "1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print("Ratio penalty curve (rupees per lost fight):")
    for r in (0.000, 0.002, 0.004, 0.006, 0.008, 0.0085, 0.010, 0.012):
        print(f"  ratio {r:6.2%}  ->  {ratio_penalty(r):10,.0f}")

    # --- the same alert, two merchants ------------------------------------
    alert = _alert("DEMO-1", "CUST-DEMO", 12_000.0)
    healthy = new_state(chargebacks=160, transactions=40_000)    # 0.40%
    stressed = new_state(chargebacks=340, transactions=40_000)   # 0.85%
    a = decide(alert, 0.75, healthy, day=0)
    b = decide(alert, 0.75, stressed, day=0)

    print("\nSame alert (₹12,000, P(win) 0.75), two merchants:")
    for label, r in (("0.40% ratio", a), ("0.85% ratio", b)):
        print(f"  {label}: penalty {r['ratio_penalty']:>9,.0f} | "
              f"EV fight {r['ev_fight']:>10,.0f} vs refund {r['ev_refund']:>9,.0f} "
              f"-> {r['final_action']}")
    assert a["final_action"] == "fight", "healthy merchant should fight this"
    assert b["final_action"] == "refund", "stressed merchant should deflect this"

    # --- a customer asking for a third refund -----------------------------
    state = new_state(chargebacks=160, transactions=40_000)
    print("\nSame customer, three small alerts EV says refund:")
    for i in range(3):
        record = decide(_alert(f"DEMO-{i}", "CUST-REPEAT", 900.0), 0.20, state, day=i * 10)
        apply_outcome(state, record, day=i * 10, won=None)
        print(f"  attempt {i + 1}: {record['final_action']:6} | gates passed "
              f"{len(record['gates_passed'])}/{len(record['gates_checked'])} | {record['reason']}")
    assert state["deflections"]["CUST-REPEAT"] == [0, 10], "should have spent exactly two"

    third = decide(_alert("DEMO-4", "CUST-REPEAT", 900.0), 0.20, state, day=30)
    assert third["ev_decision"] == "refund", "EV should still want to refund"
    assert third["final_action"] == "fight", "gates should have vetoed the third"
    assert "lifetime_budget" not in third["gates_passed"]

    # --- a contradiction tightens marginal cases and only marginal cases --
    fresh = new_state(chargebacks=160, transactions=40_000)
    marginal = _alert("DEMO-C", "CUST-FIRST", 25_000.0)   # EV margin ~268, under one step
    clean = decide(marginal, 0.20, fresh, day=0)
    flagged = decide(marginal, 0.20, fresh, day=0, llm={"has_internal_contradiction": True})
    assert clean["final_action"] == "refund" and flagged["final_action"] == "fight", \
        "contradiction should tip a marginal first deflection to fight"
    assert "contradictory complaint" in flagged["reason"], flagged["reason"]
    clear = decide(_alert("DEMO-E", "CUST-FIRST", 5_000.0), 0.05, fresh, day=0,
                   llm={"has_internal_contradiction": True})
    assert clear["final_action"] == "refund", \
        "a clear refund case must survive a contradiction flag"
    failed_ex = decide(marginal, 0.20, fresh, day=0,
                       llm={"has_internal_contradiction": False, "llm_status": "api_error: x"})
    assert failed_ex["final_action"] == clean["final_action"], \
        "an LLM failure changed a decision"

    # --- kill switch and the confidence band ------------------------------
    killed = decide(alert, 0.75, healthy, day=0, kill_switch=True)
    assert killed["final_action"] == "queue", "kill switch must stop automatic action"
    coin_flip = decide(alert, 0.50, healthy, day=0)
    assert coin_flip["final_action"] == "queue", "coin-flip band must route to a human"
    for p in (0.0, 1.0):
        r = decide(alert, p, healthy, day=0)
        assert 0.0 < r["p_win"] < 1.0, "saturated probability reached the record unclamped"

    # --- the whole test split ---------------------------------------------
    from features import share_counts
    from model import predict_win_prob

    _, _, test = evaluate.load()
    p = predict_win_prob(test.drop(columns=evaluate.HIDDEN),
                         shares=share_counts(pd.read_csv(evaluate.ALERTS)))
    batch_state = new_state(chargebacks=160, transactions=40_000)
    records = run_batch(test, p, batch_state)

    counts = records["final_action"].value_counts()
    queued = counts.get("queue", 0) / len(records)
    print(f"\nTest split ({len(records)} alerts), starting ratio 0.40%:")
    for action, n in counts.items():
        print(f"  {action:7} {n:4d}  ({n / len(records):.1%})")
    print(f"  ratio moved 0.400% -> {current_ratio(batch_state):.3%}")
    print(f"  vetoed by a gate: {int((records.ev_decision != records.final_action).sum())}")
    print(f"\nhuman queue: {queued:.1%} of the test set")

    # --- costing the queue, so this is comparable to evaluate.py -----------
    cost = bracket_cost(test, records)
    baselines = {name: fn(test.drop(columns=evaluate.HIDDEN))
                 for name, fn in evaluate.STRATEGIES.items()
                 if name in ("system", "always_refund")}
    scored = {name: evaluate.score(f, test)["net_per_1000"] for name, f in baselines.items()}

    print(f"\nNet ₹ per 1000 alerts, all {cost['n']} rows, flat RATIO_PENALTY yardstick:")
    print(f"  decide.py, queue resolved perfectly   {cost['best']:>12,.0f}   <- optimistic bound")
    print(f"  decide.py, queue refunded wholesale   {cost['worst']:>12,.0f}   <- pessimistic bound")
    print(f"  evaluate.py `system` (no queue)       {scored['system']:>12,.0f}")
    print(f"  always_refund                         {scored['always_refund']:>12,.0f}")
    print(f"  review cost included in both bounds: {cost['review_cost']:,} "
          f"({cost['queued']} alerts x {HUMAN_REVIEW_COST})")

    # The same batch charged the penalty the rule itself believed at decision
    # time. Not comparable to the rows above -- shown so the gap between the
    # yardstick and the rule's own assumptions is visible rather than buried.
    own = bracket_cost(test, records, penalty=records["ratio_penalty"].to_numpy())
    print(f"  ...same batch under its own dynamic penalty: "
          f"{own['worst']:,.0f} to {own['best']:,.0f}")

    assert cost["best"] >= cost["worst"], "bracket is inverted"
    assert cost["worst"] > scored["always_refund"], "worse than refunding everything"
    assert len(records) == len(test), "lost rows in the batch"
    assert records["p_win"].between(EPS, 1 - EPS).all(), "unclamped probability in the ledger"


if __name__ == "__main__":
    main()
