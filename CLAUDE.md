\# Pre-Dispute Deflection Shield



\## What this is

A system that intercepts pre-dispute alerts (Ethoca/Verifi-style) and decides,

per case, whether to refund immediately or prepare to fight the chargeback.

Decision is expected-value based and adjusts to how close the merchant is to

card-network monitoring thresholds.



Submission for the Razorpay AI Buildathon, Track 02 (AI Risk Manager).

Judged on: problem taste, build quality, appropriate AI use, failure recovery.



\## Hard rules

\- Deterministic logic decides money. The LLM only extracts features from

&#x20; unstructured text. Never let an LLM output a refund/fight decision directly.

\- Every money action goes through policy.py and lands in the audit log.

\- No new dependencies beyond pandas, scikit-learn, anthropic, streamlit.

&#x20; Ask me before adding anything.

\- Fewest files possible. No abstractions with one implementation, no config

&#x20; for values that never change, no scaffolding "for later".

\- Prefer stdlib. Prefer boring. I have to defend every line out loud to a

&#x20; panel of engineers.

\- Non-trivial logic leaves one runnable assert-based check behind. No pytest

&#x20; fixtures, no test suites.

\- predict\_win\_prob can return exactly 0.0 or 1.0 (isotonic saturates).

&#x20; Guard any log, ratio, or division that consumes it.

\- Self-check runs use DISABLE\_LLM=1. Live API calls only when explicitly

&#x20; scoring the LLM.



\## Vocabulary

\- alert: pre-dispute notification, arrives before a formal chargeback

\- deflect: refund now so the dispute never formalises

\- represent: fight the chargeback with an evidence packet

\- ratio: chargebacks / total transactions. Above \~1% the merchant enters a

&#x20; card-network monitoring programme (fines, then loss of card processing).

\- P(win): model output — probability we win this dispute if we fight it



\## Style

\- Type hints on function signatures. No classes unless state genuinely persists.

\- Comments explain \*why\*, never \*what\*.

\- Mark deliberate corner-cuts with `# ponytail:` naming the ceiling and upgrade path.


## Commits
- Commit messages describe what changed, never when. No day numbers.
- Use conventional prefixes: feat, fix, refactor, docs, test, chore.
- One logical unit of work per commit. Don't batch unrelated changes.
- Example: `feat: EV decision engine with dynamic ratio penalty`
