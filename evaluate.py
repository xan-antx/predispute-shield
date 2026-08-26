"""Measurement harness. Every later change is judged against this file.

Built against dummy strategies on purpose: if the scoreboard only appears once a
model exists, the model's number has nothing to beat and "it looks good" becomes
the acceptance test. always_refund is the bar to clear -- deflecting everything
is what a cautious merchant actually does today.

Strategies never see the label, the persona, or the contradiction flag. They get
exactly the columns a real alert carries at decision time.
"""

import random
import sys
from pathlib import Path

import pandas as pd

ALERTS = Path("data/alerts.csv")
RESULTS = Path("results.md")

TEST_FRAC = 0.20
CALIB_FRAC = 0.20
SEED = 42

REPRESENT_COST = 400    # analyst time plus gateway fee to assemble and file an evidence packet
FEE_SAVED = 1200        # network chargeback fee the merchant never pays if the dispute is deflected
RATIO_PENALTY = 2000    # ratio cost of the chargeback a fight lets formalise.
                        # Charged on EVERY fight, won or lost: under VAMP the
                        # dispute counts from filing, and winning the representment
                        # later does not take it back off the ratio. VAMP is
                        # count-based, so a flat per-event penalty is the correct
                        # shape, not a simplification.
                        # ponytail: flat here. The real penalty is a step function of how close
                        # the merchant is to the 1.5% threshold -- decide.py makes it dynamic.

# Ground truth and analysis-only columns. Never handed to a strategy.
HIDDEN = ["would_win_if_fought", "persona", "ring_archetype", "text_contradiction"]


def load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """60/20/20 train / calibration / test.

    Three ways, not two, because the calibration correction in model.py is
    itself fitted on data. Fitting it on the scored rows made its own Brier look
    perfect while estimating nothing -- it was learning that holdout's noise.
    The calibration slice pays for an honest number on test.

    Test is carved out first so it stays identical regardless of what happens
    upstream of it, and every strategy here is scored on that slice alone."""
    df = pd.read_csv(ALERTS)
    test = df.sample(frac=TEST_FRAC, random_state=SEED)
    rest = df.drop(test.index)
    calibration = rest.sample(frac=CALIB_FRAC / (1 - TEST_FRAC), random_state=SEED)
    return rest.drop(calibration.index), calibration, test


# Strategies. Each takes the visible alert columns and returns a boolean Series:
# True = fight the chargeback, False = refund now and deflect it.

def always_fight(alerts: pd.DataFrame) -> pd.Series:
    return pd.Series(True, index=alerts.index)


def always_refund(alerts: pd.DataFrame) -> pd.Series:
    return pd.Series(False, index=alerts.index)


def coin_flip(alerts: pd.DataFrame) -> pd.Series:
    rng = random.Random(SEED)
    return pd.Series([rng.random() < 0.5 for _ in alerts.index], index=alerts.index)


def threshold_2000(alerts: pd.DataFrame) -> pd.Series:
    """The rule most merchants already run: refund small, fight big."""
    return alerts["amount"] >= 2000


def system(alerts: pd.DataFrame) -> pd.Series:
    """Expected value on the calibrated P(win) from model.py.

        EV(refund) = -amount + FEE_SAVED
        EV(fight)  = -REPRESENT_COST - RATIO_PENALTY - (1 - p) * amount

    The ratio penalty sits outside the (1 - p) term: filing the representment is
    what lets the chargeback formalise, so the ratio damage is paid win or lose.
    Fight only when EV(fight) is strictly larger; rearranged, that means fighting
    once p * amount exceeds FEE_SAVED + REPRESENT_COST + RATIO_PENALTY. Below
    that sum in amount, no probability can justify a fight.
    """
    # Imported inside the function: model.py imports this module for the split,
    # so a top-level import would close the cycle.
    from features import share_counts
    from model import predict_win_prob

    # Share counts come from the whole file, matching how the model was trained.
    # A 600-row slice on its own would report every ring account as unshared.
    p = predict_win_prob(alerts, shares=share_counts(pd.read_csv(ALERTS)))
    return fight_ev_rule(alerts["amount"], p)


def fight_ev_rule(amount: pd.Series, p) -> pd.Series:
    """The EV comparison itself, shared with sweep.py so a money-model change
    cannot quietly fork between the leaderboard and the sweep."""
    ev_refund = -amount + FEE_SAVED
    ev_fight = -REPRESENT_COST - RATIO_PENALTY - (1 - p) * amount
    return ev_fight > ev_refund


STRATEGIES = {
    "always_fight": always_fight,
    "always_refund": always_refund,
    "random": coin_flip,
    "threshold_2000": threshold_2000,
    "system": system,
}


def score(fight: pd.Series, test: pd.DataFrame) -> dict:
    amount = test["amount"]
    won = test["would_win_if_fought"].astype(bool)

    refund_payoff = -amount + FEE_SAVED
    # The ratio penalty applies to every fight, won or lost -- the chargeback
    # counted from the moment it was filed.
    fight_payoff = pd.Series(float(-REPRESENT_COST - RATIO_PENALTY), index=test.index).where(
        won, -amount - REPRESENT_COST - RATIO_PENALTY
    )
    net = fight_payoff.where(fight, refund_payoff)

    tp = (fight & won).sum()             # fought and won
    fp = (fight & ~won).sum()            # fought and lost
    fn = (~fight & won).sum()            # refunded something we would have won

    # Cost of a wrong call = money left on the table versus the right call.
    # FP regret is flat by construction (FEE_SAVED + REPRESENT_COST + RATIO_PENALTY);
    # FN regret scales with amount, and goes negative below ~3600 because
    # deflecting a small winnable dispute genuinely beats winning it once the
    # win itself still books a chargeback against the ratio.
    fp_cost = (refund_payoff - fight_payoff)[fight & ~won].sum()
    fn_cost = (fight_payoff - refund_payoff)[~fight & won].sum()

    return {
        "rows": len(test),
        "net": net.sum(),
        "net_per_1000": net.sum() / len(test) * 1000,
        "refunded": int((~fight).sum()),
        "fought": int(fight.sum()),
        "fights_lost": int(fp),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "recall": tp / (tp + fn) if (tp + fn) else float("nan"),
        "fp_cost": fp_cost,
        "fn_cost": fn_cost,
    }


def _pct(x: float) -> str:
    return "--" if x != x else f"{x:.1%}"  # NaN when a strategy never fights


def markdown(results: dict[str, dict]) -> str:
    header = (
        "| Strategy | Net ₹ | Net ₹/1000 alerts | Refunded | Fought | Fights lost "
        "| Precision | Recall | FP cost ₹ | FN cost ₹ |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n"
    )
    rows = "".join(
        f"| `{name}` | {r['net']:,.0f} | {r['net_per_1000']:,.0f} | {r['refunded']} "
        f"| {r['fought']} | {r['fights_lost']} | {_pct(r['precision'])} | {_pct(r['recall'])} "
        f"| {r['fp_cost']:,.0f} | {r['fn_cost']:,.0f} |\n"
        for name, r in results.items()
    )
    n = next(iter(results.values()))["rows"]
    return (
        f"# Strategy comparison\n\n"
        f"{n} test alerts (20% holdout of a 60/20/20 split, seed {SEED}). "
        f"Net is rupees; less negative is better.\n"
        f"Precision/recall are for the decision to fight, scored against `would_win_if_fought`.\n"
        f"FP cost = rupees lost by fighting disputes we lost, versus deflecting them.\n"
        f"FN cost = rupees lost by refunding disputes we would have won.\n\n"
        f"{header}{rows}\n"
        f"Costs: represent {REPRESENT_COST}, fee saved {FEE_SAVED}, ratio penalty {RATIO_PENALTY}.\n"
    )


def main() -> None:
    train, calibration, test = load()
    assert (len(train), len(calibration), len(test)) == (1800, 600, 600), "split drifted"
    visible = test.drop(columns=HIDDEN)
    assert not set(HIDDEN) & set(visible.columns), "a strategy can see ground truth"

    results = {name: score(fn(visible), test) for name, fn in STRATEGIES.items()}

    assert len({r["rows"] for r in results.values()}) == 1, "strategies scored different row counts"
    assert results["always_refund"]["net"] != results["always_fight"]["net"], "money model is inert"
    assert results["always_fight"]["refunded"] == 0 and results["always_refund"]["fought"] == 0

    report = markdown(results)
    # The report carries rupee signs and the Windows console defaults to cp1252.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    RESULTS.write_text(report, encoding="utf-8")
    print(report)
    print(f"written to {RESULTS}")


if __name__ == "__main__":
    main()
