"""Parameter sweep: is the headline an artifact of one simulator configuration?

The committed result lives at one point in configuration space -- ~20% label
noise, ~53% base win rate, one log-normal amount distribution. This file
regenerates the world at 27 grid points (3 noise levels x 3 base rates x 3
amount shapes), retrains the full model pipeline per cell, and scores the same
EV strategy against always_refund on that cell's own 600-row test slice. Losing
cells are reported, not excluded: a sweep that only prints wins is marketing.

Each cell is seeded as 1000 + cell index, so any single cell is reproducible in
isolation. The canonical seed-42 dataset is regenerated at the end, and the run
aborts loudly if the restored world does not reproduce the committed results.

ponytail: cells run the real pipeline end to end (~10s each, ~5 minutes total).
A faster sweep would reuse features across label redraws; not worth the code.
"""

import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

import evaluate
import model
import simulator

SWEEP = Path("sweep.md")

NOISES = (0.10, 0.20, 0.30)
BASES = (0.35, 0.50, 0.65)
AMOUNTS = (("log-normal σ=0.85 (current)", 0.85),
           ("tight σ=0.45", 0.45),
           ("heavy σ=1.30", 1.30))


def run_cell(noise: float, base: float, sigma: float, seed: int) -> dict:
    df, probs = simulator.generate(seed=seed, amount_sigma=sigma,
                                   target_base_rate=base, target_noise=noise,
                                   return_probs=True)
    df.to_csv(evaluate.ALERTS, index=False)

    trained = model.train(include_easy=False)
    _, _, test = evaluate.load()
    fight = evaluate.fight_ev_rule(test["amount"], trained["test_p"])

    system = evaluate.score(fight, test)
    refund = evaluate.score(pd.Series(False, index=test.index), test)
    return {
        "system": system["net_per_1000"],
        "refund": refund["net_per_1000"],
        "fought": system["fought"],
        "fights_lost": system["fights_lost"],
        # Achieved, not target: the calibration is a root find, and the corner
        # cells are the ones most likely to miss. Report what actually happened.
        "achieved_base": float(np.asarray(probs).mean()),
        "achieved_noise": float(np.minimum(probs, 1 - np.asarray(probs)).mean()),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cells = []
    grid = list(product(AMOUNTS, NOISES, BASES))
    for idx, ((aname, sigma), noise, base) in enumerate(grid):
        seed = 1000 + idx
        r = run_cell(noise, base, sigma, seed)
        r.update(amount=aname, sigma=sigma, noise=noise, base=base, seed=seed,
                 delta=r["system"] - r["refund"])
        cells.append(r)
        print(f"[{idx + 1:2d}/{len(grid)}] {aname:28s} noise {noise:.2f} base {base:.2f} "
              f"-> Δ {r['delta']:>10,.0f}  (fought {r['fought']})")

    # Reproducibility: the first cell rerun from its seed must match exactly.
    again = run_cell(NOISES[0], BASES[0], AMOUNTS[0][1], seed=1000)
    assert again["system"] == cells[0]["system"] and again["refund"] == cells[0]["refund"], \
        "cell 0 did not reproduce from its seed"

    # Restore the canonical world and prove it: the committed results.md must
    # come back byte for byte, or this sweep corrupted the repo's numbers.
    simulator.generate().to_csv(evaluate.ALERTS, index=False)
    committed = evaluate.RESULTS.read_text(encoding="utf-8")
    evaluate.main()
    assert evaluate.RESULTS.read_text(encoding="utf-8") == committed, \
        "canonical results.md did not reproduce after the sweep"

    wins = [c for c in cells if c["delta"] > 0]
    losses = [c for c in cells if c["delta"] < 0]
    no_fight = [c for c in cells if c["fought"] == 0]

    grids = ""
    for aname, _ in AMOUNTS:
        grids += f"\n### {aname}\n\n| noise \\ base rate | " + \
                 " | ".join(f"{b:.2f}" for b in BASES) + " |\n" + \
                 "|---|" + "--:|" * len(BASES) + "\n"
        for noise in NOISES:
            row = [next(c for c in cells
                        if c["amount"] == aname and c["noise"] == noise and c["base"] == b)
                   for b in BASES]
            grids += f"| {noise:.2f} | " + " | ".join(
                f"{'**' if c['delta'] > 0 else ''}{c['delta']:+,.0f}"
                f"{'**' if c['delta'] > 0 else ''}" for c in row) + " |\n"

    detail = "".join(
        f"| {c['amount']} | {c['noise']:.2f} | {c['base']:.2f} | {c['achieved_noise']:.3f} "
        f"| {c['achieved_base']:.3f} | {c['system']:,.0f} | {c['refund']:,.0f} "
        f"| {c['delta']:+,.0f} | {c['fought']} | {c['fights_lost']} | {c['seed']} |\n"
        for c in cells)

    losing = "".join(
        f"- {c['amount']}, noise {c['noise']:.2f}, base {c['base']:.2f}: "
        f"Δ {c['delta']:+,.0f} ({c['fought']} fights, {c['fights_lost']} lost)\n"
        for c in sorted(losses, key=lambda c: c["delta"])) or "- none\n"

    report = (
        "# Simulator parameter sweep\n\n"
        "27 configurations: label noise x base win rate x amount distribution. Each cell\n"
        "regenerates 3,000 alerts (seed = 1000 + cell index), retrains the calibrated\n"
        "model on that world's 60/20/20 split, and scores the EV strategy against\n"
        "always_refund on the 600-row test slice. Δ = system minus always_refund in net\n"
        "₹ per 1000 alerts; positive (bold) means the system wins. Cells at or below\n"
        "the incumbent are listed, not excluded.\n"
        f"\n**System beats always_refund in {len(wins)} of {len(cells)} cells "
        f"({len(wins) / len(cells):.0%}).**\n"
        "On a 600-row slice, one flipped fight outcome moves Δ by ₹6,000 per 1000\n"
        "(the flat FP regret), so a cell within that band of zero is indistinguishable\n"
        "from always_refund at this sample size -- not a loss.\n"
        f"{grids}\n"
        "## Cells at or below the incumbent\n\n" + losing +
        f"\nCells where the system never fights (Δ exactly 0 by construction): "
        f"{len(no_fight)}.\n"
        "\n## Detail\n\n"
        "| Amounts | Noise tgt | Base tgt | Noise achieved | Base achieved "
        "| System ₹/1000 | Refund ₹/1000 | Δ | Fought | Lost | Seed |\n"
        "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|\n" + detail +
        "\nAchieved noise/base come from the calibrated generative probabilities, not\n"
        "the sampled labels; corner cells (low base + high noise) sit at the feasible\n"
        "boundary and can miss their targets -- read the achieved columns first.\n")

    SWEEP.write_text(report, encoding="utf-8")
    print(f"\n{len(wins)}/{len(cells)} cells won, {len(losses)} lost, "
          f"{len(no_fight)} never fought -> {SWEEP}")


if __name__ == "__main__":
    main()
