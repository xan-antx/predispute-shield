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

TEST_FRAC = 0.30
SEED = 42

REPRESENT_COST = 400    # analyst time plus gateway fee to assemble and file an evidence packet
FEE_SAVED = 1200        # network chargeback fee the merchant never pays if the dispute is deflected
RATIO_PENALTY = 2000    # cost of a lost fight counting toward the monitoring ratio.
                        # ponytail: flat here. The real penalty is a step function of how close
                        # the merchant is to the ~1% threshold -- decide.py makes it dynamic.

# Ground truth and analysis-only columns. Never handed to a strategy.
HIDDEN = ["would_win_if_fought", "persona", "text_contradiction"]


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """70/30 split. Only the test half is ever scored; the train half exists so
    the model in decide.py cannot accidentally be fit on scored rows."""
    df = pd.read_csv(ALERTS)
    test = df.sample(frac=TEST_FRAC, random_state=SEED)
    return df.drop(test.index), test


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
    """Placeholder. decide.py replaces the uniform draw with a calibrated P(win)
    and an expected-value comparison; the 0.5 cut is a stand-in for that."""
    rng = random.Random(SEED + 1)
    return pd.Series([rng.random() >= 0.5 for _ in alerts.index], index=alerts.index)


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
    fight_payoff = pd.Series(float(-REPRESENT_COST), index=test.index).where(
        won, -amount - REPRESENT_COST - RATIO_PENALTY
    )
    net = fight_payoff.where(fight, refund_payoff)

    tp = (fight & won).sum()             # fought and won
    fp = (fight & ~won).sum()            # fought and lost
    fn = (~fight & won).sum()            # refunded something we would have won

    # Cost of a wrong call = money left on the table versus the right call.
    # FP regret is flat by construction (FEE_SAVED + REPRESENT_COST + RATIO_PENALTY);
    # FN regret scales with amount, and goes negative below ~1600 because
    # deflecting a small winnable dispute genuinely beats winning it.
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
        f"{n} test alerts (30% holdout, seed {SEED}). Net is rupees; less negative is better.\n"
        f"Precision/recall are for the decision to fight, scored against `would_win_if_fought`.\n"
        f"FP cost = rupees lost by fighting disputes we lost, versus deflecting them.\n"
        f"FN cost = rupees lost by refunding disputes we would have won.\n\n"
        f"{header}{rows}\n"
        f"Costs: represent {REPRESENT_COST}, fee saved {FEE_SAVED}, ratio penalty {RATIO_PENALTY}.\n"
    )


def main() -> None:
    _, test = load()
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
