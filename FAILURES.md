# Failures

## Ring personas inflated to 23% of the dataset
**When:** day 1, first full run of `simulator.py`
**Symptom:** persona mix in `data/alerts.csv` came out honest 62.2%, ring 23.3%, opportunist 14.5% against a target of 75/20/5. Ring was over-represented 4.7x and opportunist was starved.
**Root cause:** the customer pool was built by drawing one persona per loop iteration with `random.choices(PERSONAS, weights=(0.75, 0.20, 0.05))`, then appending. But a ring draw does not append one account, it appends a whole cluster of 3-6 siblings sharing a device and address. So 5% of *draws* became roughly 22% of *accounts* (mean cluster size 4.5), and since alerts sample uniformly from the pool, the row-level mix inherited the same skew.
**Fix:** build the pool by count instead of by weighted draw — `round(0.75 * n)` honest, `round(0.20 * n)` opportunist, then mint ring clusters until the ring account count reaches `round(0.05 * n)`. Added a shuffle before assigning `customer_id` so the id itself carries no persona signal, and an assert in `main()` pinning the observed mix to the target.
**Result:** honest 74.9%, ring 4.9%, opportunist 20.2%. Ring cluster sizes still land at 3-6 accounts per shared device. Overall win rate moved 49.4% -> 52.0% and per-persona win rates to 55/44/42, all still inside the no-leakage band.

## Rupee sign crashed the console on Windows
**When:** day 3, first run of `evaluate.py`
**Symptom:** `UnicodeEncodeError: 'charmap' codec can't encode character '₹' in position 353`. The traceback pointed at `print(report)`, not at the file write — and `results.md` on disk was complete and correct, which made the failure look stranger than it was.
**Root cause:** the markdown table uses `₹` in its column headers. `RESULTS.write_text(..., encoding="utf-8")` handled that fine, but `print()` encodes through `sys.stdout`, which on Windows defaults to the console codepage (cp1252). cp1252 has no rupee sign, so the encode raised. Nothing to do with pandas or the report contents — purely the output stream.
**Fix:** `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` before printing, with `errors="replace"` so an exotic terminal degrades to `?` instead of taking the harness down. The file write needed no change; it was already explicitly UTF-8.
**Result:** table prints and writes identically. Worth remembering for `dashboard.py` and any evidence-packet output — every rupee amount this project prints is a latent instance of the same bug.

## Same mix bug again, one level down: ring archetypes
**When:** day 3, splitting device and address sharing into three ring archetypes
**Symptom:** target archetype mix was 40% tight / 35% dropship / 25% household. Measured over ring *accounts* it came out 42 / 46 / 12. Household, the archetype that produces the address-sharing-without-device-sharing pattern the whole change exists to create, was at less than half its intended weight.
**Root cause:** a recurrence of the day-1 class, not a new bug. The archetype was chosen once per cluster with `random.choices(weights=(0.40, 0.35, 0.25))`, but a cluster contributes 3-6 accounts, so the draw controls the mix of *clusters* and not the mix of *accounts*. With roughly twenty clusters in the pool, ordinary sampling variance in cluster sizes was enough to swing an archetype by more than a factor of two. Day 1 was the identical mistake at persona level; I re-introduced it at archetype level in the same file, nine days later, having already written it up.
**Fix:** allocate by target account count, the same shape as the persona fix — loop the archetypes, mint clusters until each one's account target is reached. Then added the assert that should have existed from the start: pin each archetype's observed account share to its target within 0.10.
**Result:** 43 / 30 / 27. Residual drift is structural, since clusters are minted whole and the last one can overshoot by up to five accounts, and the 0.10 tolerance accommodates it. Checked against the failing run: household at 0.122 is 0.128 off target, so the new assert would have fired.
**Caught by:** manual inspection, not the harness. The persona-mix assert from day 1 was passing the whole time and had nothing to say about archetypes. The lesson is not "I made the mistake twice" but that writing a failure up does not prevent it — only an assert does, and the day-1 fix stopped one level too high.
