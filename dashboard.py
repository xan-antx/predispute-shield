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
DEMO_ALERT = {"alert_id": "DEMO", "customer_id": "CUST-DEMO", "amount": 12_000.0}


@st.cache_data
def flip_point() -> float:
    """Lowest slider ratio at which the demo alert flips to refund. Computed
    from decide.decide itself so the marker can never drift from the code."""
    for pct in (i / 100 for i in range(101)):
        s = decide.new_state(chargebacks=int(pct * 10_000), transactions=1_000_000)
        if decide.decide(DEMO_ALERT, 0.75, s, day=0)["final_action"] == "refund":
            return pct
    return float("nan")


flip = flip_point()
ratio_pct = st.slider("Current chargeback ratio (%)", 0.0, 1.0, 0.40, 0.01,
                      help=f"The decision flips at {flip:.2f}%")
side = "FIGHT side" if ratio_pct < flip else "REFUND side"
st.caption(f"Flip point: {flip:.2f}% -- the slider is on the {side}.")
state = decide.new_state(chargebacks=int(ratio_pct * 10_000), transactions=1_000_000)
record = decide.decide(DEMO_ALERT, 0.75, state, day=0)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Ratio penalty", f"₹{record['ratio_penalty']:,.0f}")
c2.metric("EV(fight)", f"₹{record['ev_fight']:,.0f}")
c3.metric("EV(refund)", f"₹{record['ev_refund']:,.0f}")
colour = {"fight": "red", "refund": "green"}.get(record["final_action"], "orange")
c4.markdown(f"Action\n### :{colour}[{record['final_action'].upper()}]")
st.caption("Fixed alert: ₹12,000, P(win) 0.75, signed delivery proof. Identical "
           "evidence resolves both ways -- the penalty prices in how close the "
           "merchant is to the card-network monitoring threshold.")

# --- 3. the audit trail --------------------------------------------------
st.header("Audit trail -- last 20 decisions")
log_all = audit.read()
# The log also carries refund-execution lines (marked by an "event" field);
# this panel shows decisions, which are the rows carrying a reason.
log = (log_all[log_all["reason"].notna()].tail(20).copy()
       if "reason" in log_all.columns else log_all)
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

# --- 3b. the real refund ------------------------------------------------
st.header("Last live execution")
executions = (log_all[(log_all["event"] == "refund_execution")
                      & (log_all["execution_status"] == "refunded")]
              if "event" in log_all.columns else log_all.iloc[0:0])
if len(executions):
    live = executions.iloc[-1]
    e1, e2, e3 = st.columns(3)
    e1.metric("Provider refund id", live["provider_refund_id"])
    e2.metric("Status", str(live["execution_status"]).upper())
    e3.metric("Amount", f"₹{live['amount']:,.2f}")
    blocked = log_all[(log_all["event"] == "refund_blocked")
                      & (log_all["alert_id"] == live["alert_id"])]
    if len(blocked):
        st.caption(f"A retry of alert {live['alert_id']} was refused: "
                   f"blocked_duplicate, prior_status="
                   f"{blocked.iloc[-1]['prior_status']}. The double-refund "
                   f"guard, observed live against Razorpay test mode.")
else:
    st.caption("No live execution recorded in audit.jsonl yet.")

# --- 3c. winnable, not expensive ----------------------------------------
st.header("It picks winnable fights, not expensive ones")


@st.cache_data
def fight_comparison() -> tuple[pd.DataFrame, float]:
    _, _, test = evaluate.load()
    ours = evaluate.system(test.drop(columns=evaluate.HIDDEN))
    priciest = test["amount"].rank(ascending=False, method="first") <= int(ours.sum())
    rows, nets = [], []
    for name, fought in ((f"System: {int(ours.sum())} evidence-picked fights", ours),
                         (f"Counterfactual: the {int(priciest.sum())} most expensive", priciest)):
        nets.append(evaluate.score(fought, test)["net_per_1000"])
        rows.append({"Strategy": name, "Fought": int(fought.sum()),
                     "Win rate": f"{test.would_win_if_fought[fought].mean():.1%}",
                     "Net ₹/1000": f"{nets[-1]:,.0f}"})
    return pd.DataFrame(rows), nets[0] - nets[1]


comparison, edge = fight_comparison()
st.dataframe(comparison, use_container_width=True, hide_index=True)
st.caption(f"Same fight budget, ₹{edge:,.0f} per 1000 alerts apart: the edge is "
           f"evidence selection, not ticket size.")

# --- 4. the strategy table ----------------------------------------------
st.header("Strategy comparison")
st.markdown(Path("results.md").read_text(encoding="utf-8").split("\n", 1)[1])
