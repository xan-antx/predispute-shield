"""Append-only decision log.

One JSON object per line, opened in append mode and closed immediately. No
rewrites, no updates, no deletes -- if a decision was made it stays in the file
in the order it happened. That is the whole design. A money decision nobody can
reconstruct afterwards is a money decision nobody can defend, and the record
written here carries the inputs, the arithmetic, the gates and the reason, so a
disputed refund six months from now can be explained without rerunning anything.

JSONL rather than a table because the schema will grow -- llm.py will want to
attach what it extracted -- and appending a field to a JSON object breaks nothing
that already reads the file.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = Path("audit.jsonl")


def _plain(value):
    """numpy scalars arrive from pandas and are not JSON-serialisable. Unwrap
    rather than stringify, so the log stays typed for whoever reads it back."""
    return value.item() if hasattr(value, "item") else str(value)


def log(event: dict, path: Path = LOG) -> None:
    """Append one event. Timestamp first so the line reads chronologically."""
    record = {"ts": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=_plain) + "\n")


def read(path: Path = LOG) -> pd.DataFrame:
    """The whole log as a DataFrame, for the dashboard. Empty frame if nothing
    has been logged yet, so callers do not each need their own guard."""
    if not path.exists():
        return pd.DataFrame()
    with path.open(encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def line_count(path: Path = LOG) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> None:
    # CLAUDE.md hard rule: self-checks never make live calls. setdefault keeps
    # an explicit DISABLE_LLM=0 able to override for a deliberate live run.
    os.environ.setdefault("DISABLE_LLM", "1")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # Imported here: decide.py imports this module, and a top-level import
    # would close the cycle.
    import evaluate
    import decide
    from features import share_counts
    from model import predict_win_prob

    # Append-only means the check measures a delta. Truncating the log to get a
    # clean baseline would be the one operation this file exists to prevent.
    before = line_count()

    _, _, test = evaluate.load()
    p = predict_win_prob(test.drop(columns=evaluate.HIDDEN),
                         shares=share_counts(pd.read_csv(evaluate.ALERTS)))
    records = decide.run_batch(test, p, decide.new_state(160, 40_000))

    after = line_count()
    with LOG.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()][before:]
    parsed = [json.loads(line) for line in lines]          # raises if any line is malformed
    # The batch appends decision records plus refund-execution lines (marked by
    # an "event" field); one decision per alert, executions alongside.
    decisions = [r for r in parsed if "event" not in r]
    assert len(decisions) == len(records), f"logged {len(decisions)} of {len(records)} decisions"
    assert after - before == len(parsed), "line count disagrees with parse"

    required = {"alert_id", "p_win", "ev_fight", "ev_refund", "current_ratio", "ratio_penalty",
                "ev_decision", "gates_checked", "gates_passed", "final_action", "reason",
                "execution_status"}
    missing = required - set(decisions[0])
    assert not missing, f"decision record is missing {sorted(missing)}"
    assert all(required <= set(r) for r in decisions), "a record dropped fields mid-batch"
    refunds = [r for r in decisions if r["final_action"] == "refund"]
    assert all(r["execution_status"] != "not_applicable" for r in refunds), \
        "a refund decision skipped execution"

    df = read()
    assert len(df) == after, "read() disagrees with the file"

    # A veto has to be readable straight off the line: EV wanted one thing, a
    # gate produced another, and the failed gate is the one checked but not passed.
    fresh = pd.DataFrame(decisions)
    vetoed = fresh[fresh.ev_decision != fresh.final_action]
    assert len(vetoed), "no veto in this batch -- the check cannot prove it is visible"
    assert all(len(r.gates_passed) < len(r.gates_checked) for r in vetoed.itertuples()), \
        "a veto left no failed gate behind"
    agreed = fresh[fresh.ev_decision == fresh.final_action]
    assert all(len(r.gates_passed) == len(r.gates_checked) for r in agreed.itertuples()), \
        "an unvetoed decision shows a failed gate"

    print(f"{len(records)} decisions appended, log now {after} lines")
    print(f"vetoed by a gate: {len(vetoed)}  |  EV honoured: {len(agreed)}")
    print("\nfailed gates in this batch:")
    failed = [g for r in vetoed.itertuples() for g in set(r.gates_checked) - set(r.gates_passed)]
    for gate, n in pd.Series(failed).value_counts().items():
        print(f"  {gate:20} {n:4d}")
    print(f"\nsample line:\n  {lines[0].strip()[:200]}...")


if __name__ == "__main__":
    main()
