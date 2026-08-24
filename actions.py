"""Execution layer: the point where a refund decision becomes money moving.

Everything upstream simulates; this file calls Razorpay's test-mode refund API.
Two disciplines carry over from llm.py and one is new:

- Failure returns structure, never raises. Timeout, non-2xx, malformed body all
  come back as a dict with execution_status; the batch continues.
- Circuit breaker after CIRCUIT_LIMIT consecutive failures.
- Idempotency through the audit log. An "initiated" line is appended BEFORE the
  network call, and the guard blocks any alert whose LAST execution status is
  anything except a definite rejection. A timeout is not a failure -- the refund
  may have landed -- so it blocks forever and is released only by manual
  reconciliation against the provider dashboard. A crash between call and
  result leaves "initiated" as the last word, which also blocks. The single
  retryable state is "failed": the provider answered and said no, no money
  moved.

DRY_RUN=1 (the default) skips the network but walks the full guard-and-log
path, so batch runs are safe and the idempotency machinery is exercised by
every run. Live calls only with an explicit DRY_RUN=0.

ponytail: the guard rescans audit.jsonl per call, O(file) each time -- fine at
this scale. A real deployment puts a unique constraint on alert_id in a refunds
table and lets the database enforce this. alert_id also travels to the provider
in notes and receipt, so the other side of a disputed double-refund is
searchable in the Razorpay dashboard.
"""

import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

import audit
from llm import _load_dotenv

RAZORPAY_BASE = "https://api.razorpay.com/v1"
TIMEOUT_SECONDS = 15.0
CIRCUIT_LIMIT = 5
RETRYABLE = ("failed",)   # provider said no; every other prior status blocks

_consecutive_failures = 0


def _dry_run() -> bool:
    return os.environ.get("DRY_RUN", "1") != "0"


def _credentials() -> tuple[str, str] | None:
    _load_dotenv()
    key_id = os.environ.get("RAZORPAY_KEY_ID")
    secret = os.environ.get("RAZORPAY_KEY_SECRET")
    return (key_id, secret) if key_id and secret else None


def _post(url: str, payload: dict, auth: tuple[str, str]) -> dict:
    """One POST. Raises on anything that is not a parsed 2xx body."""
    token = base64.b64encode(f"{auth[0]}:{auth[1]}".encode()).decode()
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={
        "Authorization": f"Basic {token}",
        "Content-Type": "application/json",
        "User-Agent": "predispute-shield/1.0",
    })
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as resp:
        return json.load(resp)


def _last_status(alert_id: str) -> str | None:
    """Last execution status for this alert in the append-only log."""
    if not audit.LOG.exists():
        return None
    last = None
    with audit.LOG.open(encoding="utf-8") as f:
        for line in f:
            if '"refund_execution"' not in line:
                continue
            rec = json.loads(line)
            if rec.get("event") == "refund_execution" and rec.get("alert_id") == alert_id:
                last = rec.get("execution_status")
    return last


def execute_refund(payment_id: str, amount: float, alert_id: str) -> dict:
    """Refund `amount` rupees on `payment_id`, at most once per alert_id ever.
    Never raises."""
    global _consecutive_failures
    result = {"alert_id": alert_id, "payment_id": payment_id, "amount": amount,
              "provider_refund_id": None}

    prior = _last_status(alert_id)
    if prior is not None and prior not in RETRYABLE:
        audit.log({"event": "refund_blocked", "alert_id": alert_id,
                   "payment_id": payment_id, "prior_status": prior})
        return {**result, "execution_status": "blocked_duplicate", "prior_status": prior}

    if _consecutive_failures >= CIRCUIT_LIMIT:
        return {**result, "execution_status": "circuit_open"}

    if _dry_run():
        audit.log({"event": "refund_execution", "alert_id": alert_id,
                   "payment_id": payment_id, "amount": amount,
                   "execution_status": "dry_run"})
        return {**result, "execution_status": "dry_run"}

    creds = _credentials()
    if creds is None:
        _consecutive_failures += 1
        audit.log({"event": "refund_execution", "alert_id": alert_id,
                   "payment_id": payment_id, "amount": amount,
                   "execution_status": "failed", "error": "razorpay credentials not set"})
        return {**result, "execution_status": "failed",
                "error": "razorpay credentials not set"}

    # Written before the network call: if the process dies mid-request, this
    # line is what stops a restart from refunding the same alert twice.
    audit.log({"event": "refund_execution", "alert_id": alert_id,
               "payment_id": payment_id, "amount": amount,
               "execution_status": "initiated"})
    payload = {"amount": int(round(amount * 100)),   # rupees -> paise
               "notes": {"alert_id": alert_id},
               "receipt": alert_id}
    try:
        body = _post(f"{RAZORPAY_BASE}/payments/{payment_id}/refund", payload, creds)
        status, extra = "refunded", {"provider_refund_id": body.get("id")}
        _consecutive_failures = 0
    except urllib.error.HTTPError as exc:
        # The provider answered and said no: no money moved, retry is safe.
        detail = exc.read().decode(errors="replace")[:200]
        status, extra = "failed", {"error": f"HTTP {exc.code}: {detail}"}
        _consecutive_failures += 1
    except Exception as exc:                          # noqa: BLE001 - never propagate
        # Anything else is ambiguous: the request may have reached the provider.
        # Both classifications block retries; the split exists for the human
        # doing reconciliation, not for the code.
        timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower()
        status = "timeout_unknown" if timed_out else "error_unknown"
        extra = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        _consecutive_failures += 1

    audit.log({"event": "refund_execution", "alert_id": alert_id,
               "payment_id": payment_id, "amount": amount,
               "execution_status": status, **extra})
    return {**result, "execution_status": status, **extra}


def main() -> None:
    global _consecutive_failures, _post
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    saved_env = {k: os.environ.get(k)
                 for k in ("DRY_RUN", "RAZORPAY_KEY_ID", "RAZORPAY_KEY_SECRET")}
    saved_post = _post
    os.environ["DRY_RUN"] = "1"
    # Unique per run: the guard is deliberately permanent across runs, so a
    # reused self-check id would block on the previous invocation's entries.
    run = int(time.time())

    a1 = f"SELF-{run}-A"
    assert execute_refund("pay_selfcheck", 500.0, a1)["execution_status"] == "dry_run"
    second = execute_refund("pay_selfcheck", 500.0, a1)
    assert second["execution_status"] == "blocked_duplicate", second
    print("double-refund guard blocks a repeat alert_id (DRY_RUN=1)")

    # Simulated live path: fake creds, transport stubbed. No network anywhere.
    os.environ.update({"DRY_RUN": "0", "RAZORPAY_KEY_ID": "rzp_test_selfcheck",
                       "RAZORPAY_KEY_SECRET": "selfcheck"})

    def timeout_post(url, payload, auth):
        raise TimeoutError("simulated: timed out")
    _post, _consecutive_failures = timeout_post, 0
    a2 = f"SELF-{run}-B"
    out = execute_refund("pay_selfcheck", 500.0, a2)
    assert out["execution_status"] == "timeout_unknown", out
    assert _last_status(a2) == "timeout_unknown", "timeout not logged as attempted"
    retry = execute_refund("pay_selfcheck", 500.0, a2)
    assert retry["execution_status"] == "blocked_duplicate", retry
    print("timeout logs attempted-not-completed; retry cannot double-refund")

    def dead_post(url, payload, auth):
        raise ConnectionError("simulated: connection refused")
    _post, _consecutive_failures = dead_post, 0
    for i in range(CIRCUIT_LIMIT):
        r = execute_refund("pay_selfcheck", 500.0, f"SELF-{run}-C{i}")
        assert r["execution_status"] == "error_unknown", r
    tripped = execute_refund("pay_selfcheck", 500.0, f"SELF-{run}-C{CIRCUIT_LIMIT}")
    assert tripped["execution_status"] == "circuit_open", tripped
    print(f"circuit opens after {CIRCUIT_LIMIT} consecutive failures")

    def reject_post(url, payload, auth):
        raise urllib.error.HTTPError(url, 400, "Bad Request", None,
                                     io.BytesIO(b'{"error": {"code": "BAD_REQUEST_ERROR"}}'))
    _post, _consecutive_failures = reject_post, 0
    a3 = f"SELF-{run}-D"
    assert execute_refund("pay_selfcheck", 500.0, a3)["execution_status"] == "failed"
    # A definite rejection is the one state the guard lets through again.
    assert execute_refund("pay_selfcheck", 500.0, a3)["execution_status"] == "failed"
    print("definite provider rejection stays retryable")

    _post, _consecutive_failures = saved_post, 0
    for key, value in saved_env.items():
        os.environ.pop(key, None) if value is None else os.environ.update({key: value})
    print("all execution self-checks pass, no network touched")


if __name__ == "__main__":
    main()
