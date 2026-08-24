# Failures

## Ring personas inflated to 23% of the dataset
**When:** day 1, first full run of `simulator.py`
**Symptom:** persona mix in `data/alerts.csv` came out honest 62.2%, ring 23.3%, opportunist 14.5% against a target of 75/20/5. Ring was over-represented 4.7x and opportunist was starved.
**Root cause:** the customer pool was built by drawing one persona per loop iteration with `random.choices(PERSONAS, weights=(0.75, 0.20, 0.05))`, then appending. But a ring draw does not append one account, it appends a whole cluster of 3-6 siblings sharing a device and address. So 5% of *draws* became roughly 22% of *accounts* (mean cluster size 4.5), and since alerts sample uniformly from the pool, the row-level mix inherited the same skew.
**Fix:** build the pool by count instead of by weighted draw — `round(0.75 * n)` honest, `round(0.20 * n)` opportunist, then mint ring clusters until the ring account count reaches `round(0.05 * n)`. Added a shuffle before assigning `customer_id` so the id itself carries no persona signal, and an assert in `main()` pinning the observed mix to the target.
**Result:** honest 74.9%, ring 4.9%, opportunist 20.2%. Ring cluster sizes still land at 3-6 accounts per shared device. Overall win rate moved 49.4% -> 52.0% and per-persona win rates to 55/44/42, all still inside the no-leakage band.
