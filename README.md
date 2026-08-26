# Pre-Dispute Deflection Shield

Razorpay AI Buildathon, Track 02 (AI Risk Manager).

**The finding this system is built on:** under the card networks' own
accounting, a chargeback counts against the merchant's monitoring ratio from
the moment it is filed — winning the representment later does not take it back
off. It follows that there exist disputes worth refunding **even at P(win) =
1.0**: the ratio slot is spent whether you win or lose, and only a refund in
the pre-dispute window prevents the dispute from ever existing. That is a
consequence of network rules, not of this simulator. Everything below is
machinery for pricing that trade per alert.

## The problem

When a customer disputes a charge, their bank sends the merchant a pre-dispute
alert (Ethoca/Verifi-style) before the formal chargeback is filed. That window —
minutes to hours — is the only point where a refund makes the dispute vanish
entirely: no fee, no ratio hit, no evidence packet. Almost nobody uses it well,
because no merchant can staff a per-alert judgement call that has to happen in
minutes, so the real choice today is a blanket rule: refund everything or fight
everything. Razorpay's Agent Studio Dispute Responder answers disputes that
already exist; this system decides whether a dispute is permitted to exist at
all.

## Why the window is worth more than the fee

Card networks put merchants into monitoring programmes when the chargeback
ratio crosses roughly 1% of transactions: fines first, then mandated
remediation, then loss of card processing. So the cost of losing one dispute is
not flat — it rises steeply as the merchant approaches the line. This system
prices that in with a convex penalty (`ratio_penalty` in
[decide.py](decide.py)): ₹896 per lost fight at 20% of the threshold, ₹30,899
at 85%. The same alert, same evidence, resolves *fight* for a healthy merchant
and *refund* for a stressed one, and that is correct behaviour, not
inconsistency. The specific constants — floor ₹500, ceiling ₹50,000, cubic
shape — are illustrative; the shape is the claim, the magnitudes are stand-ins
for an acquirer's real fee schedule.

## The complaint is written by the adversary

The one free-text input in a dispute -- the complaint -- is authored by the
person the merchant may be about to fight. Any model that reads it can be
steered by rewording it, so this system is built around one rule: **a model
that reads attacker-controlled text must not control money**. The LLM here
extracts three fields from complaint text and none of them enters the EV
arithmetic or the feature matrix; its single, measured route to a decision is
bounded at ~0.8% of net (details in the AI section below), an outage degrades
to exactly the no-LLM system, and the customer-supplied complaint category is
excluded from the win-probability model outright -- measured to cost
approximately nothing. The same logic decides what the model may see: the
feature split in [features.py](features.py) is hard-to-fake evidence versus
customer-authored claims, and money rides only on the first.

## Results

Every number below is negative by construction — a pre-dispute alert always
costs money, and the game is losing less. 600 held-out alerts (20% test slice
of a 60/20/20 split, seed 42), full table in [results.md](results.md):

| Strategy | Net ₹/1000 alerts | Fought | Fights lost | Precision | Recall |
|---|--:|--:|--:|--:|--:|
| always_fight | -3,901,315 | 600 | 307 | 48.8% | 100% |
| always_refund | -1,816,286 | 0 | 0 | — | 0% |
| random | -2,911,061 | 296 | 156 | 47.3% | 47.8% |
| threshold_2000 | -2,376,247 | 297 | 140 | 52.9% | 53.6% |
| **system** | **-1,588,913** | **60** | **15** | **75.0%** | 15.4% |

Against the incumbent behaviour — deflect everything, which is what a cautious
merchant actually does, and which beats every naive alternative on this board —
the system saves roughly **₹227k per 1000 alerts** (₹227,373 on this split;
treat the last digits as noise). That figure was ~₹514k before the ratio
accounting was corrected to VDMP-style — a chargeback counts from filing, win
or lose — which made fighting costlier everywhere and cut the claimed edge
roughly in half; the correction was kept and the constants were not retuned to
win the number back. Recall is low and deliberately so: below ₹3,600 no win
probability justifies fighting, because even a won fight books the chargeback
against the ratio, so deflecting a small winnable dispute beats winning it.

## It finds a winnable subpopulation, not the expensive ones

The system fights 60 of 600 alerts with mean predicted P(win) 0.782 and an
**actual win rate of 0.750**, against a base rate of 48.8% — the calibration
holds on rows the model never saw, exactly where money is committed. The
obvious null hypothesis is that it just fights the big-ticket alerts. It
doesn't: fighting the 60 *most expensive* alerts instead wins only 48.3% and
loses ₹127,027 more per 1000, and only 34 of the two sets of 60 overlap.
Amount matters — the EV threshold falls as amount rises, by design — but
evidence selection is doing the majority of the work.

## Does it survive other worlds?

Two sweeps check whether the conclusion is an artifact of one configuration.
[sweep.md](sweep.md) regenerates the world at 27 points (label noise × base
win rate × amount distribution) and retrains per cell: the system beats
always_refund in 26; in the 27th (tight amounts, 30% noise, 35% base rate) it
fought once, lost that fight, and landed ₹6,000 per 1000 below the incumbent —
exactly one flipped outcome, indistinguishable from always_refund at this
sample size. [penalty_sweep.md](penalty_sweep.md) varies the invented penalty
constants (floor × ceiling × exponent): 15 wins, 9 exact ties where the curve
prices every fight out and the system degenerates to the incumbent, and 3
cells within one to two flipped outcomes below it. Across all 54 worlds the
upside reaches +₹1.7M per 1000 and the worst cell is a scratch inside the
noise band: when there is winnable volume to find, the system finds it, and
when there is none it degrades to the incumbent rather than below it.

## The model is at the noise floor

Labels are sampled from a sigmoid over evidence, so even an oracle that knows
the generative weights exactly cannot score better than the irreducible noise.
That oracle scores Brier 0.1425 on this test slice; the shipped model scores
0.1469 (ROC-AUC 0.862) — about 3% above a perfect oracle, and crucially **not
below it**, which is what feature leakage would look like. The model is a
calibrated logistic regression whose coefficients recover the simulator's
evidence weights in the correct order. Details in
[calibration.md](calibration.md).

Be clear about what this result is: a pipeline-correctness check, not a
modelling achievement. The simulator generates labels from a sigmoid over
linear evidence weights, and the model is a logistic regression over those
same evidence features — correctly specified for the data-generating process
by construction, so landing near the floor is close to guaranteed. What the
number actually proves is the absence of two failure modes: no feature
leakage (which would put the model *below* the floor) and no pipeline bugs
(which would put it meaningfully above). On real data, where nobody hands you
the generative model, the gap to the floor would be real and unknown.

## Where the AI is, and where it isn't

Deterministic expected-value arithmetic decides money. The LLM
([llm.py](llm.py)) reads complaint text and returns three fields; none enters
the feature matrix or the EV terms. Its single route to a decision is bounded
and measured: an internal contradiction in the complaint raises the required
EV margin by ₹500 in one policy gate, so it can tip cases whose margin is
under that — 4 of 600 decisions when the flag is forced on for *every* alert,
about 0.8% of net. (Under the pre-VDMP money model this was 31 decisions and
0.1%: correcting the ratio accounting widened most refund margins past the
gate's reach but raised the stakes of each remaining flip.) `results.md` is checksum-identical with `DISABLE_LLM=1`,
and the whole system runs with no key at all: every LLM failure mode returns
neutral defaults, chosen so an outage can never cost a customer a refund.

Measured against the simulator's ground truth (100-row stratified sample, two
independent runs): contradiction detection recall **0.949** in both, precision
1.000 — but on ~45 negatives, where a single flipped case would drop derived
precision to ~0.75, so read precision as "high", not as certainty.

## The policy layer costs money, and that's the point

The full decision path — dynamic ratio penalty, deflection budgets, velocity
caps, a kill switch, and a human queue for P(win) between 0.40 and 0.60 —
lands between ₹76k and ₹112k per 1000 *behind* the raw EV strategy, depending
on how well humans resolve the 60 queued alerts (10% of the batch). That cost
is accepted, not hidden: EV optimises the batch you can see, and the gates
protect against the batch you can't — the serial refund farmer, the model gone
stale, the coin-flip case nobody should automate. Every decision lands in an
append-only audit log ([audit.py](audit.py)) carrying the inputs, the
arithmetic, the gates checked and the reason, written before the outcome is
known.

## Why you should distrust these numbers

The data is synthetic: 3,000 alerts from [simulator.py](simulator.py), seeded
(`random.seed(42)`, `np.random.seed(42)`), with labels sampled from a sigmoid
over hand-chosen evidence weights. That bakes in the central assumptions —
which evidence matters, how much irreducible noise exists (~15–20%), an 8%
band of genuinely undecidable cases — so the model is partly recovering a
world I built. The complaint texts are templated, so the 0.949 contradiction
recall is an upper bound; human-written complaints would score lower. The test
slice is 600 rows: a decile of it is 60 alerts with a 95% interval around
±0.13, so third-decimal differences here are not findings.

As evidence that these caveats are enforced rather than decorative: an earlier
calibration correction measured well on a two-way split, was rebuilt under an
honest three-way split, failed to reproduce, and was removed — the full
post-mortem is in [FAILURES.md](FAILURES.md), alongside four other failures.
Known unfixed gaps are in [LIMITATIONS.md](LIMITATIONS.md).

## Why not X

**Why no agent framework?** Money decisions need to be replayable and
defensible line by line. A deterministic EV function plus explicit gates is
auditable; an agent loop is not.

**Why logistic regression over gradient boosting?** The rule, set before
looking: switch only if boosting wins on *both* Brier and AUC. It won on
neither, which the noise-floor analysis explains — there is almost nothing
left to extract, so the model with readable coefficients keeps the slot.

**Why doesn't the LLM decide anything?** See "The complaint is written by
the adversary" above — it is the design rule the whole system hangs off.

**Why stdlib urllib instead of the openai package?** Groq's endpoint is one
JSON POST. Adding a dependency for that violates the project's own rules more
than it helps.

## Architecture

- [simulator.py](simulator.py) generates 3,000 alerts with evidence-only
  sampled labels; persona and ring structure shape the *distributions*, never
  the label.
- [features.py](features.py) splits features into hard-to-fake (delivery
  proof, account history, device/address share counts) and easy-to-fake
  (customer-supplied category, excluded from the money path).
- [model.py](model.py) trains a calibrated P(win) on the 60% train slice,
  selects on the 20% calibration slice, reports only on the untouched 20% test.
- [decide.py](decide.py) computes EV with the dynamic ratio penalty, then runs
  the policy gates, routes coin-flips to a human queue, and prices the queue
  as a bracket.
- [audit.py](audit.py) appends every decision to `audit.jsonl` before the
  outcome is known; [llm.py](llm.py) feeds one bounded policy signal in.

## How this was built

Implementation was written with Claude Code; architecture, the money model,
and every judgement call about signal versus noise were mine. The recurring
lesson of the build, documented across [FAILURES.md](FAILURES.md): three times
a rule existed in prose but not in code — the ring archetype mix, the review
harness's silent zero-verdict run, the DISABLE_LLM convention — and each time
the failure recurred or nearly did until an assert made it mechanical. A
written rule prevents nothing; only an enforced one does.

## How to run

```bash
pip install -r requirements.txt
python simulator.py     # regenerates data/alerts.csv (seeded, deterministic)
python model.py         # trains, writes model.pkl and calibration.md
python evaluate.py      # strategy comparison, writes results.md
python decide.py        # EV + gates demo and batch self-check
python audit.py         # audit-log self-check
```

Self-check entrypoints set `DISABLE_LLM=1` themselves, so bare runs never make
API calls. For live LLM extraction and contradiction scoring, put
`GROQ_API_KEY=...` in `.env` (gitignored) and run `python llm.py`; the
provider and model are switchable via `LLM_PROVIDER` and `GROQ_MODEL`.
