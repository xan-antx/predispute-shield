# Limitations

Things this build does not establish, written down so nobody has to discover
them by reading the code carefully. `FAILURES.md` covers what broke and got
fixed; this file covers what is still true.

## The deflection gates barely fire on this dataset

`decide.py` runs three gates on any decision to refund: a lifetime deflection
budget, a 90-day velocity cap, and an escalating EV margin that rises with each
prior deflection. Across the 600-row test split they veto **4 decisions** — three
on the lifetime budget, one on the escalating margin. The velocity cap never
fires at all.

This is a property of the data, not a defect in the gates. The test slice holds
434 customers who appear exactly once, 69 who appear twice, and 9 who appear
three or more times. A lifetime budget of 2 has almost nothing to bind on. The
gates are demonstrably reachable — `decide.py`'s self-check constructs a customer
who asks for three refunds and asserts the third is refused, and that assert
fails if the gate is broken — but "reachable in a constructed case" is a weaker
claim than "load-bearing in production", and only the first is supported here.

Stressing them properly needs `simulator.py` to concentrate alerts on repeat
customers rather than sampling uniformly from the customer pool: ring accounts
especially, since serial deflection is the abuse pattern the budget exists to
stop, and right now a ring account is no likelier to file a second alert than
anyone else. Until that changes, treat the gates as designed and tested but not
yet exercised.

## The human queue is bracketed, not measured

10% of the test split (60 alerts) routes to a human on the low-confidence band.
Nobody resolves them, so their cost is reported as a range: an analyst who calls
every one correctly, and an analyst who refunds all of them. The true number is
between those two and this dataset cannot narrow it. `HUMAN_REVIEW_COST` is
illustrative.

## The ratio penalty constants are invented

The shape of `ratio_penalty` — convex, biting well before the threshold — is the
claim worth defending. The floor of 500, the ceiling of 50,000 and the cubic
exponent are plausible stand-ins, not sourced from an acquirer's fee schedule.
Any rupee figure downstream inherits that.

The directional error this section used to flag -- only lost fights counting
toward the ratio -- has been corrected: every fight now books the chargeback
from filing, win or lose, in both the money model and the merchant state. The
correction cut the headline saving from ~₹514k to ~₹227k per 1000 alerts and
was kept without retuning any constant to compensate.

## VAMP's 1,500-event floor is not modelled

VAMP only formally monitors a merchant once fraud reports plus disputes reach
1,500 combined events in a month. Below that floor the network programme does
not bite, and this system's ratio penalty overstates the network-side cost --
though acquirers run their own, stricter thresholds underneath, which is what
the penalty floor loosely stands in for. The simulated merchant here (160
chargebacks against 40,000 transactions as a starting point) is treated as
monitored; a small merchant is not, and for them the EV case for fighting is
stronger than this system computes.

## No timestamps anywhere

`simulator.py` emits no event time. Two things work around it and both are marked
`# ponytail:` in the code — device and address share counts are computed over the
whole dataset, which leaks a little future information into every training row,
and `run_batch` spreads a batch evenly across a year so the velocity window has
something to measure. One field in the simulator fixes both.

## Sample size

Test is 600 rows. A decile of it is 60 alerts, and a 60-row bucket at p≈0.5
carries a 95% interval of roughly ±0.13. Differences in the third or fourth
decimal of Brier, AUC or win rate are not findings at this size — that mistake is
already written up in `FAILURES.md` and the same caution applies to every number
in `calibration.md` and `results.md`.

## Full-amount refunds 400 in Razorpay test mode

Across six live test-mode attempts, every refund of the entire remaining
balance of a payment (₹10 of ₹10; ₹8 when ₹2 was already refunded) returned
Razorpay's generic `400 BAD_REQUEST_ERROR: invalid request sent`, while every
partial refund of the same payment, through the same code path, with the same
field shape, succeeded. The omit-amount hypothesis — Razorpay defaults an amount-less
refund to the full balance, so leaving the field out might dodge the
validation — was then tested live and refuted: the same generic 400 comes
back. Seven attempts total: three full-remaining refunds failed (explicit
₹10, explicit ₹8, amount omitted with ₹4 remaining), five partial refunds
succeeded. The quirk is refunding a payment to zero, however expressed, and
its mechanism is invisible from outside. A production deployment refunds
whole payments as its normal case, so it would hit this immediately; settle
it with Razorpay support before relying on `execute_refund` for full refunds.

## TC40 asymmetry: deflection saves the dispute, not the fraud report

A pre-dispute refund prevents the TC15 dispute from ever existing, and VAMP
excludes RDR/CDRN-resolved cases from its numerator — but if the cardholder's
bank already filed a TC40 fraud report, that event is in the numerator
regardless of what the merchant does next. Deflection cannot un-file it. For
fraud-coded alerts, then, this model **overstates deflection's ratio benefit**:
it credits every refund with saving a full ratio slot, when an
unauthorised-category alert has often already spent one via the TC40. The bias
direction is consistent — the system looks slightly better than it is exactly
on the alert category where fraud rings live, which tilts EV toward refunding
them.

The fix is known and bounded: discount the ratio component of `FEE_SAVED` for
unauthorised-category alerts by the probability that a TC40 was already filed.
It is not implemented because the simulator does not model TC40 filing at all,
so the discount would be an invented parameter dressed up as a correction —
worse than an acknowledged bias, because it would look measured. On real data
the TC40 feed exists and the discount becomes an observable rate rather than a
guess.
