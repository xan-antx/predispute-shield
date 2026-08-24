"""The only AI in the system. It reads text; deterministic logic decides.

Three structured fields out of free-form complaint text, with one bounded route
to a decision. complaint_category is already excluded from features.py;
specificity_score is reporting only; has_internal_contradiction feeds exactly
one policy gate in decide.py, where it raises the required EV margin by one
step (₹500) -- enough to tip a deflection whose margin is under that, and
measured at 31 of 600 decisions (~0.1% of net) when forced on for every alert.
It never enters ev_fight, ev_refund, p_win or the feature matrix. If this file
returned "refund" it would be a different system with a different risk profile,
and the argument for it would have to be made to a regulator rather than a
judge.

Everything here is written to fail into neutrality. Timeout, malformed JSON, API
error, missing key, degenerate response shapes, no credentials at all -- every
path returns the same neutral defaults and the batch continues. The defaults are
chosen so a failure never counts against a customer: has_internal_contradiction
False means an LLM outage cannot cost anyone their refund. Set DISABLE_LLM=1 to
skip the call entirely; the system runs fully without it, and evaluate.py
produces identical numbers either way because none of this reaches the decision
arithmetic.

The complaint text is written by the disputing customer, who in this system may
be a fraudster. It is treated as hostile input end to end: capped in length
before it buys tokens, never able to widen its own influence past the three
fields, and never echoed uncapped into the audit log.
"""

import json
import os
import re
import sys

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 300
TIMEOUT_SECONDS = 12.0
MAX_TEXT_CHARS = 4000     # complaints fit in a fraction of this; the cap exists so
                          # a hostile customer cannot buy an arbitrary token bill
CIRCUIT_LIMIT = 5         # consecutive API errors before we stop calling at all
CATEGORIES = ("non_receipt", "not_as_described", "unauthorised", "other")

# Neutral until proven otherwise. "other" commits to nothing, 0.5 is the middle
# of the specificity range, and False is the answer that cannot penalise anyone.
DEFAULTS = {
    "complaint_category": "other",
    "has_internal_contradiction": False,
    "specificity_score": 0.5,
    "llm_status": "default",
}

PROMPT = """Extract three fields from this customer dispute complaint.

complaint_category: one of non_receipt, not_as_described, unauthorised, other
has_internal_contradiction: true if the complaint contradicts itself. The clearest
  case is a customer claiming they never received an item while describing that
  item's condition, colour, fit or behaviour -- you cannot describe what never
  arrived. Also true for a customer disputing a charge as unauthorised while
  describing the product they received. Do not mark ordinary vagueness,
  frustration or poor grammar as contradiction.
specificity_score: 0.0 to 1.0. Vague complaints with no verifiable detail score
  low. Complaints naming dates, order contents, tracking events or prior contact
  score high.

Respond with one JSON object and nothing else. No prose, no markdown fences.
{{"complaint_category": "...", "has_internal_contradiction": true|false, "specificity_score": 0.0}}

Complaint:
{text}{transcript}"""

STRICTER = """Your previous response could not be parsed as JSON.

Output exactly one JSON object. First character must be {{. Last character must
be }}. No explanation before or after it. No markdown code fences. Use only these
keys: complaint_category, has_internal_contradiction, specificity_score.

Complaint:
{text}{transcript}"""

_client = None
_unavailable_reason: str | None = None
_consecutive_api_errors = 0


def _get_client():
    """Build the client once. A missing key is a normal operating condition here,
    not an error: the system is designed to run without this file working."""
    global _client, _unavailable_reason
    # Reason first: once set (missing creds, open circuit) it wins even if a
    # client object was already built.
    if _unavailable_reason is not None:
        return None
    if _client is not None:
        return _client

    try:
        import anthropic
    except Exception as exc:                      # noqa: BLE001 - a broken or
        # version-conflicting install raises more than ImportError at import time
        _unavailable_reason = f"anthropic import failed: {type(exc).__name__}"
        return None
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        # Checked up front so a 600-alert batch does not make 600 doomed calls
        # and wait out 600 timeouts to reach the same defaults.
        # ponytail: the SDK can also resolve an `ant auth login` profile or
        # workload identity federation; this guard covers only the env vars.
        # Anyone on profile auth can export ANTHROPIC_AUTH_TOKEN to get past it.
        _unavailable_reason = "ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN not set"
        return None
    try:
        # max_retries=1 covers a transient 429 or 5xx. Parse failures are handled
        # separately below, because the fix for those is a different prompt.
        _client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=1)
    except Exception as exc:                      # noqa: BLE001 - never propagate
        _unavailable_reason = f"client init failed: {type(exc).__name__}"
        return None
    return _client


def _parse(raw: str) -> dict:
    """Pull a JSON object out of the response and coerce it into range.

    Tolerates a fenced or prose-wrapped object because that is the cheap failure
    to absorb; anything else raises and the caller retries once with a stricter
    prompt. Every field is validated -- a model that returns a category outside
    the enum, or a specificity of 7, is malformed even though the JSON parsed.
    Error messages truncate model-supplied values: they end up in llm_status,
    which lands in the append-only audit log, and the model's output is steerable
    by whoever wrote the complaint.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in response")
    data = json.loads(match.group(0))

    category = data["complaint_category"]
    if category not in CATEGORIES:
        raise ValueError(f"category {str(category)[:40]!r} outside the enum")
    contradiction = data["has_internal_contradiction"]
    if not isinstance(contradiction, bool):
        raise ValueError(f"has_internal_contradiction is {type(contradiction).__name__}")
    score = float(data["specificity_score"])      # OverflowError on absurd ints;
    if not 0.0 <= score <= 1.0:                   # the caller's catch covers it
        raise ValueError(f"specificity {score} outside [0, 1]")

    return {
        "complaint_category": category,
        "has_internal_contradiction": contradiction,
        "specificity_score": score,
    }


def extract_features(complaint_text: str, chat_transcript: str | None = None) -> dict:
    """Three structured fields from unstructured text. Never raises.

    Returns the neutral defaults on any failure, with `llm_status` naming what
    happened so the audit log records that the LLM was down rather than silently
    recording a confident-looking "other".
    """
    global _unavailable_reason, _consecutive_api_errors
    # Checked per call, not only at client construction: DISABLE_LLM is the off
    # switch, and an off switch that only works before first use is not one.
    if os.environ.get("DISABLE_LLM") == "1":
        return {**DEFAULTS, "llm_status": "skipped: DISABLE_LLM=1"}
    client = _get_client()
    if client is None:
        return {**DEFAULTS, "llm_status": f"skipped: {_unavailable_reason}"}

    text = ("" if complaint_text is None else str(complaint_text))[:MAX_TEXT_CHARS]
    transcript = (f"\n\nChat transcript:\n{str(chat_transcript)[:MAX_TEXT_CHARS]}"
                  if chat_transcript else "")
    attempts = (
        ("ok", PROMPT.format(text=text, transcript=transcript)),
        ("ok_retry", STRICTER.format(text=text, transcript=transcript)),
    )

    last = "unknown"
    for status, prompt in attempts:
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system="You extract structured fields from dispute complaints. "
                       "You reply with one JSON object and no other text.",
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as exc:                  # noqa: BLE001 - see module docstring
            # Deliberately broad. Timeout, rate limit, auth, connection, a 500,
            # or something the SDK has not invented yet all mean the same thing
            # to this system: carry on without the LLM.
            _consecutive_api_errors += 1
            if _consecutive_api_errors >= CIRCUIT_LIMIT and _unavailable_reason is None:
                # Circuit breaker: a dead API should cost the batch a handful of
                # timeouts, not one per alert. Same reasoning as the up-front
                # credential check, applied to the outage it could not foresee.
                _unavailable_reason = f"circuit open after {CIRCUIT_LIMIT} consecutive API errors"
            return {**DEFAULTS, "llm_status": f"api_error: {type(exc).__name__}"}

        try:
            # Inside the guard on purpose: the SDK deserialises non-strictly, so
            # a degenerate body can hand back content=None or a text block whose
            # text is None, and both must degrade, never raise.
            raw = "".join(b.text for b in response.content if b.type == "text")
            result = {**_parse(raw), "llm_status": status}
            _consecutive_api_errors = 0
            return result
        except (ValueError, KeyError, TypeError, AttributeError,
                OverflowError, json.JSONDecodeError) as exc:
            # Only a parse failure retries, and only once. Retrying an API error
            # is the SDK's job; retrying a parse failure needs a firmer prompt.
            # str(exc) is capped: it can quote model output, which is steerable
            # by the complaint author, and this string lands in the audit log.
            last = f"{type(exc).__name__}: {str(exc)[:80]}"

    return {**DEFAULTS, "llm_status": f"parse_failed: {last}"}


def score_contradictions(n: int = 100, seed: int = 42) -> dict:
    """Precision and recall of contradiction detection against the simulator.

    Stratified 50/50 rather than a random 100. At the dataset's ~6% contradiction
    rate a random sample of 100 carries about six positives, and a recall
    estimate off six cases is not worth the API calls. Balancing buys a usable
    recall number, at the cost of a precision figure that no longer reflects
    deployment -- so the true-prevalence precision is derived from the measured
    rates and reported alongside it.

    Failed extractions are excluded, not scored: a failure returns the neutral
    default, which is correct for the money path and poison for measurement --
    counting it as a genuine "no contradiction" call deflates recall and
    inflates precision in exact proportion to how broken the API was that day.
    """
    import pandas as pd

    alerts = pd.read_csv("data/alerts.csv")
    prevalence = alerts["text_contradiction"].mean()
    positives = alerts[alerts.text_contradiction].sample(n // 2, random_state=seed)
    negatives = alerts[~alerts.text_contradiction].sample(n - n // 2, random_state=seed)
    sample = pd.concat([positives, negatives]).sample(frac=1, random_state=seed)

    extracted = [extract_features(t) for t in sample["complaint_text"]]
    scored = [(e["has_internal_contradiction"], t)
              for e, t in zip(extracted, sample["text_contradiction"].tolist())
              if e["llm_status"].startswith("ok")]
    excluded = len(extracted) - len(scored)

    tp = sum(p and t for p, t in scored)
    fp = sum(p and not t for p, t in scored)
    fn = sum(not p and t for p, t in scored)
    tn = sum(not p and not t for p, t in scored)

    recall = tp / (tp + fn) if tp + fn else float("nan")          # = TPR
    fpr = fp / (fp + tn) if fp + tn else float("nan")
    precision = tp / (tp + fp) if tp + fp else float("nan")
    # Bayes on the measured rates, at the rate contradictions actually occur.
    denominator = prevalence * recall + (1 - prevalence) * fpr
    true_precision = prevalence * recall / denominator if denominator else float("nan")

    return {
        "n": len(scored), "excluded_failures": excluded,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision_balanced": precision, "recall": recall, "fpr": fpr,
        "prevalence": prevalence, "precision_at_prevalence": true_precision,
    }


def main() -> None:
    global _client, _unavailable_reason, _consecutive_api_errors
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    from types import SimpleNamespace as NS

    # --- whatever credentials exist, the shape holds ----------------------
    result = extract_features("Order never arrived. The box was torn when I opened it.")
    assert set(result) == set(DEFAULTS), f"field set drifted: {sorted(result)}"
    assert result["complaint_category"] in CATEGORIES
    assert isinstance(result["has_internal_contradiction"], bool)
    assert 0.0 <= result["specificity_score"] <= 1.0
    live_ok = result["llm_status"].startswith("ok")
    print(f"live call: {result['llm_status']}")

    # --- _parse rejects malformed, absorbs fenced -------------------------
    for bad in ('not json at all', '{"complaint_category": "banana"}', '{}',
                '{"complaint_category": "other", "has_internal_contradiction": "yes",'
                ' "specificity_score": 0.5}',
                '{"complaint_category": "other", "has_internal_contradiction": false,'
                ' "specificity_score": 7}',
                '{"complaint_category": "other", "has_internal_contradiction": false,'
                ' "specificity_score": ' + "9" * 400 + "}"):
        try:
            _parse(bad)
            raise AssertionError(f"_parse accepted malformed input: {bad[:60]}")
        except (ValueError, KeyError, TypeError, OverflowError, json.JSONDecodeError):
            pass
    fenced = _parse('```json\n{"complaint_category": "non_receipt", '
                    '"has_internal_contradiction": true, "specificity_score": 0.8}\n```')
    assert fenced["has_internal_contradiction"] is True

    # --- hostile response shapes, proven without a key by stubbing the ----
    # client directly. Saved and restored so live scoring below still works.
    saved = (_client, _unavailable_reason, _consecutive_api_errors)
    saved_env = os.environ.pop("DISABLE_LLM", None)

    def stub(*responses):
        seq = iter(responses)
        return NS(messages=NS(create=lambda **kw: next(seq)))

    good = ('{"complaint_category": "non_receipt", '
            '"has_internal_contradiction": true, "specificity_score": 0.9}')
    hostile = [
        NS(content=None),                                # SDK non-strict deserialisation
        NS(content=[NS(type="text", text=None)]),        # text block without text
        NS(content=[NS(type="thinking")]),               # no text blocks at all
        NS(content=[NS(type="text", text='{"complaint_category": "other", '
                       '"has_internal_contradiction": false, "specificity_score": '
                       + "9" * 400 + "}")]),             # float() overflow after clean JSON
        NS(content=[NS(type="text", text='{"complaint_category": "'
                       + "INJECTED " * 40 + '", "has_internal_contradiction": false, '
                       '"specificity_score": 0.5}')]),   # steered text aimed at the audit log
    ]
    for resp in hostile:
        _client, _unavailable_reason, _consecutive_api_errors = stub(resp, resp), None, 0
        out = extract_features("x")
        assert out["llm_status"].startswith("parse_failed"), out
        assert out["has_internal_contradiction"] is False
        assert len(out["llm_status"]) <= len("parse_failed: ") + 100, "status not capped"
    print(f"{len(hostile)} hostile response shapes -> neutral defaults, none raised")

    _client, _unavailable_reason, _consecutive_api_errors = \
        stub(NS(content=[NS(type="text", text="no json here")]),
             NS(content=[NS(type="text", text=good)])), None, 0
    out = extract_features("x")
    assert out["llm_status"] == "ok_retry" and out["complaint_category"] == "non_receipt"
    print("stricter-prompt retry recovers a parse failure")

    def dead(**kw):
        raise ConnectionError("api down")
    _client, _unavailable_reason, _consecutive_api_errors = NS(messages=NS(create=dead)), None, 0
    for _ in range(CIRCUIT_LIMIT):
        assert extract_features("x")["llm_status"].startswith("api_error")
    out = extract_features("x")
    assert out["llm_status"].startswith("skipped: circuit open"), out
    print(f"circuit opens after {CIRCUIT_LIMIT} consecutive API errors")

    _client, _unavailable_reason = stub(NS(content=[NS(type="text", text=good)])), None
    os.environ["DISABLE_LLM"] = "1"
    assert extract_features("x")["llm_status"] == "skipped: DISABLE_LLM=1", \
        "DISABLE_LLM ignored once a client exists"
    os.environ.pop("DISABLE_LLM")
    print("DISABLE_LLM=1 honoured mid-process, cached client or not")

    _client, _unavailable_reason, _consecutive_api_errors = saved
    if saved_env is not None:
        os.environ["DISABLE_LLM"] = saved_env

    # --- live scoring, only when a real call succeeded above --------------
    if not live_ok:
        print(f"\nno live scoring: {_unavailable_reason or 'DISABLE_LLM=1'}")
        print("degradation paths verified; set ANTHROPIC_API_KEY and rerun for precision/recall")
        return

    stats = score_contradictions()
    print(f"\ncontradiction detection, {stats['n']} scored (stratified 50/50, "
          f"{stats['excluded_failures']} failed extractions excluded):")
    print(f"  tp {stats['tp']}  fp {stats['fp']}  fn {stats['fn']}  tn {stats['tn']}")
    print(f"  recall               {stats['recall']:.3f}")
    print(f"  precision (balanced) {stats['precision_balanced']:.3f}")
    print(f"  false positive rate  {stats['fpr']:.3f}")
    print(f"  precision at the real {stats['prevalence']:.1%} rate: "
          f"{stats['precision_at_prevalence']:.3f}")
    if stats["fp"] == 0:
        print("  note: zero false positives in 50 negatives puts the derived "
              "precision at exactly 1.0 -- read it as 'high', not as certainty")


if __name__ == "__main__":
    main()
