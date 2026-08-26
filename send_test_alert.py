"""Fire one sample pre-dispute alert at the webhook and print the decision.

The payload is the enriched shape webhook.py expects: the Ethoca-style dispute
fields plus the merchant's own order-history enrichment. Strong evidence on
purpose (signed proof, delivered, long history) so the demo shows a FIGHT;
change delivery_proof to "none" to watch it flip to a refund.
"""

import json
import sys
import urllib.request

ALERT = {
    "alert_id": "WEBHOOK-DEMO-1",
    "customer_id": "CUST-LIVE-1",
    "payment_id": "pay_TTgyzBMGOx7Efo",
    "amount": 12_000.0,
    "account_age_days": 900,
    "prior_orders": 34,
    "prior_disputes": 0,
    "delivery_status": "delivered",
    "delivery_proof": "signed",
    "device_fingerprint": "dev_webhookdemo1",
    "address_hash": "addr_webhookdemo1",
    "days_since_purchase": 9,
    "complaint_category": "item_not_received",
    "complaint_text": "Order never arrived. Tracking shows delivered but nothing reached me.",
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    request = urllib.request.Request(
        "http://localhost:8600/alert",
        data=json.dumps(ALERT).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        print(json.dumps(json.load(response), indent=2, default=str))


if __name__ == "__main__":
    main()
