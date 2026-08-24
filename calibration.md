# Calibration

900 held-out alerts, deciles of predicted P(win). Split and rows are identical to `evaluate.py`.

## Cost of dropping customer-supplied features

The money path excludes the easy-to-fake block (`complaint_category` one-hots). A claimant writes that field, so a model that leans on it can be moved by rewording a complaint. This is what the exclusion costs:

| Feature set | ROC-AUC | Brier | Features |
|---|--:|--:|--:|
| hard-to-fake only (shipped) | 0.8624 | 0.1465 | 15 |
| all features | 0.8628 | 0.1461 | 19 |
| **delta** | **-0.0004** | **+0.0004** | **-4** |

Reproduce with `python model.py --with-easy`.

## Applied correction

Raw predictions are mapped through the observed rate for their bucket before being returned by `predict_win_prob`. Bucket rates pass through a weighted isotonic step first, so the mapping cannot invert the ranking.

| Bucket | n | Model says | Actually won | Returned |
|---|--:|--:|--:|--:|
| (-0.001, 0.0792] | 93 | 0.034 | 0.043 | 0.043 |
| (0.0792, 0.136] | 97 | 0.112 | 0.155 | 0.126 |
| (0.136, 0.183] | 86 | 0.161 | 0.093 | 0.126 |
| (0.183, 0.511] | 84 | 0.300 | 0.226 | 0.226 |
| (0.511, 0.655] | 90 | 0.602 | 0.456 | 0.456 |
| (0.655, 0.731] | 98 | 0.701 | 0.714 | 0.714 |
| (0.731, 0.781] | 84 | 0.759 | 0.798 | 0.793 |
| (0.781, 0.871] | 109 | 0.841 | 0.789 | 0.793 |
| (0.871, 0.897] | 69 | 0.883 | 0.899 | 0.887 |
| (0.897, 0.967] | 90 | 0.916 | 0.878 | 0.887 |

## Chosen: logistic

Rule: default to logistic regression; switch only if gradient boosting wins on Brier *and* AUC.

| Model | Brier (lower better) | ROC-AUC |
|---|--:|--:|
| logistic | 0.1465 | 0.862 |
| gradient_boosting | 0.1501 | 0.858 |
| constant 0.5 | 0.2500 | 0.500 |

## Per-model detail

### logistic

precision 0.748 | recall 0.902 | F1 0.818 | ROC-AUC 0.862 | Brier 0.1465

| Predicted P(win) bucket | n | Predicted mean | Actual win rate | Gap |
|---|--:|--:|--:|--:|
| (-0.001, 0.0792] | 93 | 0.034 | 0.043 | +0.009 |
| (0.0792, 0.136] | 97 | 0.112 | 0.155 | +0.043 |
| (0.136, 0.183] | 86 | 0.161 | 0.093 | -0.068 |
| (0.183, 0.511] | 84 | 0.300 | 0.226 | -0.074 |
| (0.511, 0.655] | 90 | 0.602 | 0.456 | -0.146 |
| (0.655, 0.731] | 98 | 0.701 | 0.714 | +0.014 |
| (0.731, 0.781] | 84 | 0.759 | 0.798 | +0.038 |
| (0.781, 0.871] | 109 | 0.841 | 0.789 | -0.052 |
| (0.871, 0.897] | 69 | 0.883 | 0.899 | +0.016 |
| (0.897, 0.967] | 90 | 0.916 | 0.878 | -0.038 |

### gradient_boosting

precision 0.742 | recall 0.900 | F1 0.814 | ROC-AUC 0.858 | Brier 0.1501

| Predicted P(win) bucket | n | Predicted mean | Actual win rate | Gap |
|---|--:|--:|--:|--:|
| (-0.001, 0.0637] | 95 | 0.038 | 0.063 | +0.025 |
| (0.0637, 0.107] | 87 | 0.086 | 0.069 | -0.017 |
| (0.107, 0.203] | 88 | 0.157 | 0.170 | +0.013 |
| (0.203, 0.525] | 90 | 0.317 | 0.222 | -0.095 |
| (0.525, 0.676] | 91 | 0.610 | 0.516 | -0.094 |
| (0.676, 0.72] | 94 | 0.704 | 0.702 | -0.002 |
| (0.72, 0.798] | 85 | 0.766 | 0.741 | -0.025 |
| (0.798, 0.873] | 95 | 0.842 | 0.789 | -0.053 |
| (0.873, 0.894] | 85 | 0.883 | 0.871 | -0.013 |
| (0.894, 1.0] | 90 | 0.918 | 0.878 | -0.040 |

