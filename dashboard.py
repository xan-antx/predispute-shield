"""Demo dashboard. Exists to be filmed, not to be a product.

Four panels: the money headline, the ratio flip, the audit trail, the strategy
table. Every number is produced by the same functions that produced the
committed results -- nothing here re-derives arithmetic, so the dashboard
cannot drift from the system it is showing.
"""

import os

os.environ.setdefault("DISABLE_LLM", "1")   # fast load, no live calls

from pathlib import Path

import pandas as pd
import streamlit as st

import audit
import decide
import evaluate
from evaluate import FEE_SAVED
from features import share_counts
from model import predict_win_prob

st.set_page_config(page_title="Pre-Dispute Deflection Shield", layout="wide")
st.title("Pre-Dispute Deflection Shield")


@st.cache_data
def batch() -> pd.DataFrame:
    """One decision run over the 600-alert test slice. Cached so slider moves
    do not re-run it (or re-append 600 audit lines per interaction)."""
    _, _, test = evaluate.load()
    p = predict_win_prob(test.drop(columns=evaluate.HIDDEN),
                         shares=share_counts(pd.read_csv(evaluate.ALERTS)))
    return decide.run_batch(test, p, decide.new_state(160, 40_000))


records = batch()
refunds = records[records.final_action == "refund"]

# --- 1. the trade, stated as a pair -------------------------------------
# Avoided: the chargeback fee each deflection dodges, plus the expected ratio
# damage a lost fight would have added. Conceded: the refunds paid to get that.
avoided = FEE_SAVED * len(refunds) + ((1 - refunds.p_win) * refunds.ratio_penalty).sum()
conceded = refunds.amount.sum()
left, right = st.columns(2)
left.metric("Fees + ratio damage avoided", f"₹{avoided:,.0f}")
right.metric("Revenue voluntarily conceded", f"₹{conceded:,.0f}")
st.caption(f"{len(refunds)} deflections across the {len(records)}-alert test batch.")

# --- 2. the same alert against a moving ratio ---------------------------
st.header("Same alert, different merchant")
ratio_pct = st.slider("Current chargeback ratio (%)", 0.0, 1.0, 0.40, 0.01)
state = decide.new_state(chargebacks=int(ratio_pct * 10_000), transactions=1_000_000)
record = decide.decide({"alert_id": "DEMO", "customer_id": "CUST-DEMO", "amount": 5000.0},
                       0.75, state, day=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ratio penalty", f"₹{record['ratio_penalty']:,.0f}")
c2.metric("EV(fight)", f"₹{record['ev_fight']:,.0f}")
c3.metric("EV(refund)", f"₹{record['ev_refund']:,.0f}")
c4.metric("Action", record["final_action"].upper())
st.caption("Fixed alert: ₹5,000, P(win) 0.75, signed delivery proof. Identical "
           "evidence flips from FIGHT to REFUND near 0.55% -- the penalty prices "
           "in how close the merchant is to the card-network monitoring threshold.")

# --- 3. the audit trail --------------------------------------------------
st.header("Audit trail -- last 20 decisions")
log = audit.read().tail(20).copy()
log["vetoed"] = log["ev_decision"] != log["final_action"]
for col in ("gates_checked", "gates_passed"):
    log[col] = log[col].apply(", ".join)
show = log[["ts", "alert_id", "amount", "p_win", "ev_decision", "final_action",
            "vetoed", "gates_checked", "gates_passed", "reason"]]


def mark_vetoed(row: pd.Series) -> list[str]:
    style = "background-color: rgba(220, 60, 60, 0.30)" if row["vetoed"] else ""
    return [style] * len(row)


st.dataframe(show.style.apply(mark_vetoed, axis=1), use_container_width=True, hide_index=True)
st.caption("Highlighted rows: a policy gate overruled the EV decision. Every row "
           "is appended to audit.jsonl before the outcome is known.")

# --- 4. the strategy table ----------------------------------------------
st.header("Strategy comparison")
st.markdown(Path("results.md").read_text(encoding="utf-8").split("\n", 1)[1])
