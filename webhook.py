"""Live demo path: an Ethoca-shaped alert in, a decided-and-executed answer out.

POST /alert runs one alert through the exact production path -- predict_win_prob
-> llm.extract_features -> decide.decide -> audit.log -> actions.execute_refund
-- and returns the full decision record. Merchant state persists across requests,
so firing several alerts moves the ratio and the deflection history live.

This is a demo skin over the same functions the harness measures, not a second
implementation: nothing here touches evaluate.py, the model, or any committed
number. A real Ethoca payload carries only the dispute; the merchant enriches it
with order history before deciding. Here the sender supplies the enriched row
(send_test_alert.py shows the shape), which is the production integration point.

Run:  python webhook.py            (DRY_RUN=1 default: no live refunds)
      python send_test_alert.py    (in another terminal)
"""

import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer

import pandas as pd

import actions
import audit
import decide
import evaluate
import llm
from features import share_counts
from model import predict_win_prob

PORT = 8600
REQUIRED = ("alert_id", "customer_id", "payment_id", "amount", "account_age_days",
            "prior_orders", "prior_disputes", "delivery_status", "delivery_proof",
            "device_fingerprint", "address_hash", "days_since_purchase",
            "complaint_category", "complaint_text")

SHARES = share_counts(pd.read_csv(evaluate.ALERTS))   # ledger stand-in, loaded once
STATE = decide.new_state(chargebacks=160, transactions=40_000)
DAY = 0


def handle_alert(alert: dict) -> dict:
    global DAY
    missing = [k for k in REQUIRED if k not in alert]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    alert["amount"] = float(alert["amount"])

    # A fingerprint the ledger has never seen counts as its own first sighting.
    shares = {key: (s if alert[col] in s.index else pd.concat([s, pd.Series({alert[col]: 1})]))
              for (key, col), s in zip([("device", "device_fingerprint"),
                                        ("address", "address_hash")],
                                       [SHARES["device"], SHARES["address"]])}
    p = float(predict_win_prob(pd.DataFrame([alert]), shares=shares)[0])
    extracted = llm.extract_features(alert["complaint_text"])

    DAY += 1
    record = decide.decide(alert, p, STATE, day=DAY, llm=extracted)
    if record["final_action"] == "refund":
        execution = actions.execute_refund(str(alert["payment_id"]),
                                           record["amount"], record["alert_id"])
        record["execution_status"] = execution["execution_status"]
        record["provider_refund_id"] = execution["provider_refund_id"]
    else:
        record["execution_status"], record["provider_refund_id"] = "not_applicable", None
    audit.log(record)
    won = bool(alert.get("would_win_if_fought")) if record["final_action"] == "fight" else None
    decide.apply_outcome(STATE, record, DAY, won)
    return record


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/alert":
            return self._reply(404, {"error": "POST /alert is the only route"})
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            record = handle_alert(json.loads(body))
            self._reply(200, record)
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            self._reply(400, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})
        except Exception as exc:                      # noqa: BLE001 - a demo server
            self._reply(500, {"error": f"{type(exc).__name__}: {str(exc)[:200]}"})

    def _reply(self, status: int, payload: dict) -> None:
        data = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    mode = "DRY_RUN" if os.environ.get("DRY_RUN", "1") != "0" else "LIVE refunds"
    print(f"listening on http://localhost:{PORT}/alert  ({mode})")
    HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
