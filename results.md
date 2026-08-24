# Strategy comparison

900 test alerts (30% holdout, seed 42). Net is rupees; less negative is better.
Precision/recall are for the decision to fight, scored against `would_win_if_fought`.
FP cost = rupees lost by fighting disputes we lost, versus deflecting them.
FN cost = rupees lost by refunding disputes we would have won.

| Strategy | Net ₹ | Net ₹/1000 alerts | Refunded | Fought | Fights lost | Precision | Recall | FP cost ₹ | FN cost ₹ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `always_fight` | -2,430,479 | -2,700,532 | 0 | 900 | 421 | 53.2% | 100.0% | 1,515,600 | 0 |
| `always_refund` | -1,572,903 | -1,747,669 | 900 | 0 | 0 | -- | 0.0% | 0 | 658,024 |
| `random` | -1,958,231 | -2,175,813 | 462 | 438 | 201 | 54.1% | 49.5% | 723,600 | 319,753 |
| `threshold_2000` | -1,625,943 | -1,806,604 | 452 | 448 | 219 | 51.1% | 47.8% | 788,400 | -77,335 |
| `system` | -1,918,242 | -2,131,380 | 458 | 442 | 202 | 54.3% | 50.1% | 727,200 | 276,163 |

Costs: represent 400, fee saved 1200, ratio penalty 2000.
