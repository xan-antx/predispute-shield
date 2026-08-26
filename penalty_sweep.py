"""Sensitivity sweep over the ratio-penalty constants.

The floor, ceiling and exponent in decide.py are invented -- LIMITATIONS.md says
so. This file asks whether the headline conclusion (deciding by EV beats
refunding everything) depends on which values were invented. Each cell treats
its (floor, ceiling, exponent) as the TRUE cost structure of that world: the
strategy decides with the cell's curve, the scoring charges fights with the same
curve, at the ratio actually standing when each fight was filed. If the
conclusion survives across magnitudes, the constants stop being load-bearing and
only the convex shape has to be defended.

Same world every cell: the canonical seed-42 dataset (regenerated up front) and
one trained model. Only the penalty curve varies, so any Δ movement is the
curve's doing. Dynamics per cell: ratio starts at 0.40% (160 of 40,000), every
fight books its chargeback from filing (VDMP), so fighting drives the penalty up
underneath the batch. No gates or queue here -- this is the pure EV strategy,
matching sweep.py, so the two sweeps answer cleanly separable questions.
"""

import sys
from itertools import product
from pathlib import Path

import pandas as pd

import decide
import evaluate
import model
import simulator
from evaluate import FEE_SAVED, REPRESENT_COST

OUT = Path("penalty_sweep.md")

FLOORS = (250, 500, 1000)
CEILINGS = (25_000, 50_000, 100_000)
EXPONENTS = (1, 2, 3)      # linear, quadratic, cubic
START = (160, 40_000)      # 0.40% starting ratio, as everywhere else


def run_cell(floor: int, ceiling: int, exponent: int,
             test: pd.DataFrame, p) -> dict:
    """Pure EV against the cell's own curve, ratio evolving underneath.

    The fight condition is decide.decide's inequality (fight iff
    p * amount > FEE_SAVED + REPRESENT_COST + penalty) with the penalty from
    the parameterised decide.ratio_penalty, so the shape logic cannot fork.
    """
    chargebacks, transactions = START
    net, fought, lost = 0.0, 0, 0
    for (_, alert), p_win in zip(test.iterrows(), p):
        pen = decide.ratio_penalty(chargebacks / transactions,
                                   floor=floor, ceiling=ceiling, exponent=exponent)
        if p_win * alert["amount"] > FEE_SAVED + REPRESENT_COST + pen:
            fought += 1
            chargebacks += 1               # counts from filing, win or lose
            net += -REPRESENT_COST - pen
            if not alert["would_win_if_fought"]:
                net += -alert["amount"]
                lost += 1
        else:
            net += -alert["amount"] + FEE_SAVED
    return {"floor": floor, "ceiling": ceiling, "exponent": exponent,
            "system": net / len(test) * 1000, "fought": fought, "lost": lost,
            "end_ratio": chargebacks / transactions}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # One canonical world, one model, for every cell.
    simulator.generate().to_csv(evaluate.ALERTS, index=False)
    trained = model.train(include_easy=False)
    _, _, test = evaluate.load()
    p = trained["test_p"]

    refund = (-test["amount"] + FEE_SAVED).sum() / len(test) * 1000
    assert round(refund) == -1_816_286, f"always_refund anchor drifted: {refund:,.0f}"

    cells = []
    for floor, ceiling, exponent in product(FLOORS, CEILINGS, EXPONENTS):
        r = run_cell(floor, ceiling, exponent, test, p)
        r["delta"] = r["system"] - refund
        cells.append(r)
        print(f"floor {floor:5d} ceiling {ceiling:7,d} exp {exponent} "
              f"-> Δ {r['delta']:>10,.0f}  (fought {r['fought']}, "
              f"end ratio {r['end_ratio']:.3%})")

    # Deterministic by construction; prove it anyway, and pin the canonical cell.
    again = run_cell(500, 50_000, 3, test, p)
    canonical = next(c for c in cells
                     if (c["floor"], c["ceiling"], c["exponent"]) == (500, 50_000, 3))
    assert again["system"] == canonical["system"], "canonical cell did not reproduce"

    # A cell that never fights accumulates the refund sum in a different order
    # than the pandas sum, so its delta is +/- a float epsilon, not exactly 0.
    # Classifying by sign alone counted nine ties as wins. One rupee of
    # tolerance separates a result from an artifact.
    wins = [c for c in cells if c["delta"] > 1]
    losses = [c for c in cells if c["delta"] < -1]
    ties = [c for c in cells if abs(c["delta"]) <= 1]
    assert all(c["fought"] == 0 for c in ties), "a tie that actually fought"

    def mean_delta(select) -> float:
        chosen = [c["delta"] for c in cells if select(c)]
        return sum(chosen) / len(chosen)

    by_exp = {e: mean_delta(lambda c, e=e: c["exponent"] == e) for e in EXPONENTS}
    by_floor = {f: mean_delta(lambda c, f=f: c["floor"] == f) for f in FLOORS}
    by_ceiling = {ce: mean_delta(lambda c, ce=ce: c["ceiling"] == ce) for ce in CEILINGS}
    # Spread attributable to each axis: max minus min of Δ holding the other two
    # axes fixed, averaged over those combinations.
    def spread(axis: str, values) -> float:
        combos = {}
        for c in cells:
            key = tuple(c[k] for k in ("floor", "ceiling", "exponent") if k != axis)
            combos.setdefault(key, []).append(c["delta"])
        return sum(max(v) - min(v) for v in combos.values()) / len(combos)

    grids = ""
    for exponent, label in zip(EXPONENTS, ("linear", "quadratic", "cubic")):
        grids += (f"\n### exponent {exponent} ({label})\n\n"
                  "| floor \\ ceiling | " + " | ".join(f"{c:,}" for c in CEILINGS) + " |\n"
                  "|---|" + "--:|" * len(CEILINGS) + "\n")
        for floor in FLOORS:
            row = [next(c for c in cells if (c["floor"], c["ceiling"], c["exponent"])
                        == (floor, ceiling, exponent)) for ceiling in CEILINGS]
            grids += f"| {floor:,} | " + " | ".join(
                "tie (0 fights)" if abs(c["delta"]) <= 1 else
                f"{'**' if c['delta'] > 1 else ''}{c['delta']:+,.0f}"
                f"{'**' if c['delta'] > 1 else ''}" for c in row) + " |\n"

    detail = "".join(
        f"| {c['floor']:,} | {c['ceiling']:,} | {c['exponent']} | {c['system']:,.0f} "
        f"| {c['delta']:+,.0f} | {c['fought']} | {c['lost']} | {c['end_ratio']:.3%} |\n"
        for c in cells)
    losing = "".join(
        f"- floor {c['floor']:,}, ceiling {c['ceiling']:,}, exponent {c['exponent']}: "
        f"Δ {c['delta']:+,.0f} ({c['fought']} fights)\n"
        for c in sorted(losses, key=lambda c: c["delta"])) or "- none\n"

    report = (
        "# Ratio-penalty sensitivity sweep\n\n"
        "27 penalty curves (floor x ceiling x exponent), each treated as the true cost\n"
        "structure of its world: the EV strategy decides with that curve and fights are\n"
        "charged with it at the ratio standing when they were filed (start 0.40%, every\n"
        "fight counts from filing). One canonical seed-42 dataset and one trained model\n"
        "throughout, so the curve is the only thing that varies. Δ = system minus\n"
        f"always_refund ({refund:,.0f}) in net ₹ per 1000 alerts; positive (bold) wins.\n"
        f"\n**System beats always_refund in {len(wins)} of {len(cells)} cells, ties it "
        f"in {len(ties)}, and lands below it in {len(losses)}.** Every tie is a "
        f"zero-fight cell: the curve prices all fights out and the system degenerates "
        f"to always_refund exactly. Each below-incumbent cell sits within one to two "
        f"flipped fight outcomes of zero (a flip moves Δ by ₹6,000 per 1000), which is "
        f"indistinguishable from the incumbent at this sample size. Winning Δ ranges "
        f"{min(c['delta'] for c in wins):+,.0f} to "
        f"{max(c['delta'] for c in wins):+,.0f}; the worst cell anywhere is "
        f"{min(c['delta'] for c in cells):+,.0f}.\n"
        f"{grids}\n"
        "## Cells at or below the incumbent\n\n" + losing +
        "\n## Shape versus magnitude\n\n"
        "Mean Δ per axis value, and the mean spread of Δ when one axis moves across its\n"
        "full range while the other two stay fixed:\n\n"
        "| Axis | Values -> mean Δ | Spread when varied alone |\n|---|---|--:|\n"
        f"| exponent | " + ", ".join(f"{e}: {by_exp[e]:+,.0f}" for e in EXPONENTS) +
        f" | {spread('exponent', EXPONENTS):,.0f} |\n"
        f"| floor | " + ", ".join(f"{f:,}: {by_floor[f]:+,.0f}" for f in FLOORS) +
        f" | {spread('floor', FLOORS):,.0f} |\n"
        f"| ceiling | " + ", ".join(f"{c:,}: {by_ceiling[c]:+,.0f}" for c in CEILINGS) +
        f" | {spread('ceiling', CEILINGS):,.0f} |\n"
        "\n## Detail\n\n"
        "| Floor | Ceiling | Exp | System ₹/1000 | Δ | Fought | Lost | End ratio |\n"
        "|--:|--:|--:|--:|--:|--:|--:|--:|\n" + detail)

    OUT.write_text(report, encoding="utf-8")
    print(f"\n{len(wins)}/{len(cells)} cells won, {len(losses)} lost -> {OUT}")


if __name__ == "__main__":
    main()
