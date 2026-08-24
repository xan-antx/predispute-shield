# Strategy comparison

900 test alerts (30% holdout, seed 42). Net is rupees; less negative is better.
Precision/recall are for the decision to fight, scored against `would_win_if_fought`.
FP cost = rupees lost by fighting disputes we lost, versus deflecting them.
FN cost = rupees lost by refunding disputes we would have won.

| Strategy | Net ₹ | Net ₹/1000 alerts | Refunded | Fought | Fights lost | Precision | Recall | FP cost ₹ | FN cost ₹ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `always_fight` | -2,675,765 | -2,973,072 | 0 | 900 | 449 | 50.1% | 100.0% | 1,616,400 | 0 |
| `always_refund` | -1,726,465 | -1,918,295 | 900 | 0 | 0 | -- | 0.0% | 0 | 667,101 |
| `random` | -2,173,553 | -2,415,059 | 462 | 438 | 220 | 49.8% | 48.3% | 792,000 | 322,189 |
| `threshold_2000` | -1,756,404 | -1,951,560 | 442 | 458 | 213 | 53.5% | 54.3% | 766,800 | -69,760 |
| `system` | -2,216,175 | -2,462,416 | 458 | 442 | 224 | 49.3% | 48.3% | 806,400 | 350,410 |

Costs: represent 400, fee saved 1200, ratio penalty 2000.
