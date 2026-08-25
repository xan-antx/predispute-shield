# Strategy comparison

600 test alerts (20% holdout of a 60/20/20 split, seed 42). Net is rupees; less negative is better.
Precision/recall are for the decision to fight, scored against `would_win_if_fought`.
FP cost = rupees lost by fighting disputes we lost, versus deflecting them.
FN cost = rupees lost by refunding disputes we would have won.

| Strategy | Net ₹ | Net ₹/1000 alerts | Refunded | Fought | Fights lost | Precision | Recall | FP cost ₹ | FN cost ₹ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `always_fight` | -2,340,789 | -3,901,315 | 0 | 600 | 307 | 48.8% | 100.0% | 1,105,200 | 0 |
| `always_refund` | -1,089,772 | -1,816,286 | 600 | 0 | 0 | -- | 0.0% | 0 | -145,817 |
| `random` | -1,746,637 | -2,911,061 | 304 | 296 | 156 | 47.3% | 47.8% | 561,600 | -50,552 |
| `threshold_2000` | -1,425,748 | -2,376,247 | 303 | 297 | 140 | 52.9% | 53.6% | 504,000 | -313,841 |
| `system` | -953,348 | -1,588,913 | 540 | 60 | 15 | 75.0% | 15.4% | 54,000 | -336,241 |

Costs: represent 400, fee saved 1200, ratio penalty 2000.
