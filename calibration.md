# Calibration

60/20/20 train / calibration / test, seed 42. The model is fitted on 1800 rows, model selection and the rejected correction used a separate 600, and every number in this file is measured on the remaining 600 that neither has seen. `evaluate.py` scores its strategies on that same test slice.

## Does the decile correction help? No.

It is measured here and **not applied**. `predict_win_prob` returns the raw isotonic-calibrated probability. Both rows below are scored on test, which neither the model nor the correction was fitted to:

| P(win) as returned | ROC-AUC | Brier |
|---|--:|--:|
| raw model output (shipped) | 0.8621 | 0.1469 |
| after decile correction | 0.8604 | 0.1474 |
| delta | -0.0017 | +0.0005 |

Worse on both. The correction was built to fix an apparent -0.146 gap at the decision boundary on an earlier two-way split; that gap did not reproduce (calibration +0.027, test -0.109, train -0.008, all with confidence intervals wide enough to contain each other). See FAILURES.md.

## Cost of dropping customer-supplied features

The money path excludes the easy-to-fake block (`complaint_category` one-hots). A claimant writes that field, so a model that leans on it can be moved by rewording a complaint. This is what the exclusion costs:

| Feature set | ROC-AUC | Brier | Features |
|---|--:|--:|--:|
| hard-to-fake only (shipped) | 0.8621 | 0.1469 | 15 |
| all features | 0.8628 | 0.1444 | 19 |

**The exclusion costs approximately nothing.** That is the whole claim, and the decimals above should not be read more precisely than that. Test is 600 rows: a decile of it holds 60 alerts, and a 60-row bucket at p≈0.5 carries a 95% interval of roughly ±0.13. Differences of this size between two feature sets sit inside that band. Treating a fourth-decimal gap as a finding is the mistake already written up in FAILURES.md.

Reproduce with `python model.py --with-easy`.

## The correction that was measured and rejected

Kept for reference, not applied. This is the map that would have replaced each raw prediction with the observed rate for its bucket, fitted on the calibration slice. Bucket rates pass through a weighted isotonic step, without which a higher-predicted bucket could return a lower P(win).

| Bucket | n | Model says | Actually won | Returned |
|---|--:|--:|--:|--:|
| (-0.001, 0.067] | 60 | 0.031 | 0.083 | 0.083 |
| (0.067, 0.0921] | 62 | 0.086 | 0.129 | 0.129 |
| (0.0921, 0.158] | 73 | 0.140 | 0.247 | 0.246 |
| (0.158, 0.492] | 45 | 0.344 | 0.244 | 0.246 |
| (0.492, 0.62] | 60 | 0.551 | 0.583 | 0.583 |
| (0.62, 0.741] | 65 | 0.717 | 0.631 | 0.631 |
| (0.741, 0.781] | 55 | 0.755 | 0.764 | 0.764 |
| (0.781, 0.879] | 65 | 0.829 | 0.862 | 0.862 |
| (0.879, 0.904] | 55 | 0.896 | 0.873 | 0.873 |
| (0.904, 0.954] | 60 | 0.921 | 0.900 | 0.900 |

## Chosen: logistic

Rule: default to logistic regression; switch only if gradient boosting wins on Brier *and* AUC. Decided on calibration, before test was touched.

| Model (on calibration) | Brier (lower better) | ROC-AUC |
|---|--:|--:|
| logistic | 0.1584 | 0.848 |
| gradient_boosting | 0.1609 | 0.839 |
| constant 0.5 (on test) | 0.2500 | 0.500 |

## Honest calibration on test

Raw model output, bucketed on the rows nothing was fitted to.

### logistic on test

precision 0.746 | recall 0.894 | F1 0.814 | ROC-AUC 0.862 | Brier 0.1469

| Predicted P(win) bucket | n | Predicted mean | Actual win rate | Gap |
|---|--:|--:|--:|--:|
| (-0.001, 0.0395] | 60 | 0.022 | 0.067 | +0.045 |
| (0.0395, 0.0921] | 64 | 0.077 | 0.094 | +0.017 |
| (0.0921, 0.158] | 72 | 0.137 | 0.153 | +0.016 |
| (0.158, 0.487] | 44 | 0.309 | 0.114 | -0.195 |
| (0.487, 0.643] | 60 | 0.559 | 0.450 | -0.109 |
| (0.643, 0.741] | 62 | 0.722 | 0.710 | -0.012 |
| (0.741, 0.791] | 59 | 0.761 | 0.797 | +0.036 |
| (0.791, 0.852] | 60 | 0.819 | 0.767 | -0.052 |
| (0.852, 0.904] | 62 | 0.893 | 0.871 | -0.022 |
| (0.904, 0.985] | 57 | 0.920 | 0.860 | -0.061 |

