# Strategy comparison

600 test alerts (20% holdout of a 60/20/20 split, seed 42). Net is rupees; less negative is better.
Precision/recall are for the decision to fight, scored against `would_win_if_fought`.
FP cost = rupees lost by fighting disputes we lost, versus deflecting them.
FN cost = rupees lost by refunding disputes we would have won.

| Strategy | Net ₹ | Net ₹/1000 alerts | Refunded | Fought | Fights lost | Precision | Recall | FP cost ₹ | FN cost ₹ |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| `always_fight` | -1,754,789 | -2,924,648 | 0 | 600 | 307 | 48.8% | 100.0% | 1,105,200 | 0 |
| `always_refund` | -1,089,772 | -1,816,286 | 600 | 0 | 0 | -- | 0.0% | 0 | 440,183 |
| `random` | -1,466,637 | -2,444,394 | 304 | 296 | 156 | 47.3% | 47.8% | 561,600 | 255,448 |
| `threshold_2000` | -1,111,748 | -1,852,913 | 303 | 297 | 140 | 52.9% | 53.6% | 504,000 | -41,841 |
| `system` | -1,411,887 | -2,353,144 | 309 | 291 | 152 | 47.8% | 47.4% | 547,200 | 215,098 |

Costs: represent 400, fee saved 1200, ratio penalty 2000.
