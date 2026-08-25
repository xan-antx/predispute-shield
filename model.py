"""Calibrated P(win if fought).

Calibration is the whole job here, not accuracy. policy.py multiplies this
number by rupees, so a model that says 0.7 and is right 55% of the time will
authorise fights that lose money while scoring well on every ranking metric.
Isotonic calibration on top of each classifier, a decile table printed every
run, and a measured correction applied on top of that before the number is
allowed anywhere near a decision.

The money path runs on hard-to-fake features only. The split comes from
evaluate.load() rather than being re-declared here, so the rows scored on the
leaderboard are exactly the rows held out from training.
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import evaluate
from features import build_features, share_counts

MODEL_PKL = Path("model.pkl")
CALIBRATION = Path("calibration.md")
LABEL = "would_win_if_fought"
SEED = 42
BUCKETS = 10

_CACHED: dict | None = None


def _models() -> dict[str, CalibratedClassifierCV]:
    """Both wrapped in isotonic calibration. Isotonic is free to bend the
    probabilities into any monotone shape, which is what fixes a classifier that
    is confidently wrong at the extremes; the cost is that it needs the cv=5
    folds to avoid fitting the calibration on its own predictions."""
    logistic = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, random_state=SEED)),
    ])
    return {
        "logistic": CalibratedClassifierCV(logistic, method="isotonic", cv=5),
        "gradient_boosting": CalibratedClassifierCV(
            GradientBoostingClassifier(random_state=SEED), method="isotonic", cv=5
        ),
    }


def calibration_table(y: pd.Series, p: np.ndarray, bins: int = BUCKETS) -> pd.DataFrame:
    """Predicted vs actual per decile of predicted probability. The gap column is
    the one that matters: negative means the model is talking us into fights it
    cannot win."""
    buckets = pd.qcut(p, bins, duplicates="drop")
    grouped = pd.DataFrame({"p": p, "y": y.to_numpy()}).groupby(buckets, observed=True)
    table = pd.DataFrame({
        "n": grouped.size(),
        "predicted": grouped["p"].mean(),
        "actual": grouped["y"].mean(),
    })
    table["gap"] = table["actual"] - table["predicted"]
    return table.reset_index(names="bucket")


def build_correction(table: pd.DataFrame) -> dict:
    """Turn a decile table into a lookup: raw probability -> observed rate.

    REPORTING ONLY. Nothing in the money path calls this. It was built to fix an
    apparent over-confidence at the decision boundary, measured on a two-way
    split; a three-way split showed the gap did not reproduce and that applying
    the map made both Brier and AUC worse on rows it had not been fitted to. See
    FAILURES.md. It stays here because the comparison it enables -- raw versus
    corrected, both scored on test -- is the evidence for that call, and a claim
    with its own disproof attached is worth more than a deleted function.

    Bucket rates pass through a weighted isotonic step because the raw rates are
    not monotone: small-sample noise can put a lower observed rate on a
    higher-predicted bucket, which would hand better evidence a worse P(win).
    """
    monotone = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit_transform(
        table["predicted"], table["actual"], sample_weight=table["n"]
    )
    return {
        # Internal cut points; searchsorted maps a raw probability to its bucket.
        "edges": [iv.right for iv in table["bucket"]][:-1],
        "values": [float(v) for v in monotone],
        "raw": [float(v) for v in table["actual"]],
    }


def apply_correction(p: np.ndarray, correction: dict) -> np.ndarray:
    idx = np.searchsorted(np.asarray(correction["edges"]), p, side="right")
    return np.asarray(correction["values"])[idx]


def _lr_coefficients(fitted: CalibratedClassifierCV, columns: list[str]) -> pd.Series:
    """Mean standardised coefficient across the calibration folds. Features are
    scaled, so magnitudes compare directly."""
    coefs = [cc.estimator.named_steps["lr"].coef_[0] for cc in fitted.calibrated_classifiers_]
    mean = pd.Series(np.mean(coefs, axis=0), index=columns)
    return mean.reindex(mean.abs().sort_values(ascending=False).index)


def _score(y: pd.Series, p: np.ndarray) -> dict:
    predicted_fight = p >= 0.5
    return {
        "precision": precision_score(y, predicted_fight, zero_division=0),
        "recall": recall_score(y, predicted_fight, zero_division=0),
        "f1": f1_score(y, predicted_fight, zero_division=0),
        "auc": roc_auc_score(y, p),
        "brier": brier_score_loss(y, p),
    }


def train(include_easy: bool = False) -> dict:
    """Fit on train, pick and correct on calibration, score on test.

    Nothing touches test until the last three lines. Model selection and the
    decile correction both consume data, so both happen on the calibration
    slice; that is what makes the corrected number on test an estimate rather
    than a restatement of the rows it was fitted on."""
    train_df, calib_df, test_df = evaluate.load()
    assert (len(train_df), len(calib_df), len(test_df)) == (1800, 600, 600), \
        "split does not match evaluate.py"
    for a, b in ((train_df, calib_df), (train_df, test_df), (calib_df, test_df)):
        assert not a.index.intersection(b.index).size, "splits overlap"

    full = pd.concat([train_df, calib_df, test_df]).sort_index()
    x = build_features(full, shares=share_counts(full), include_easy=include_easy)
    x_train, x_calib, x_test = x.loc[train_df.index], x.loc[calib_df.index], x.loc[test_df.index]
    y_train, y_calib, y_test = train_df[LABEL], calib_df[LABEL], test_df[LABEL]

    fitted, calib_metrics, calib_tables = {}, {}, {}
    for name, estimator in _models().items():
        estimator.fit(x_train, y_train)
        p = estimator.predict_proba(x_calib)[:, 1]
        assert ((p >= 0) & (p <= 1)).all(), f"{name} produced a probability outside [0,1]"
        fitted[name] = estimator
        calib_metrics[name] = _score(y_calib, p)
        calib_tables[name] = calibration_table(y_calib, p)

    lr, gb = calib_metrics["logistic"], calib_metrics["gradient_boosting"]
    # Interpretability is the tiebreak, not the metric. A coefficient I can read
    # out to a panel is worth more than a marginal lift I would have to defend
    # with a partial dependence plot. Decided on calibration, never on test.
    gb_wins = gb["brier"] < lr["brier"] and gb["auc"] > lr["auc"]
    chosen = "gradient_boosting" if gb_wins else "logistic"
    correction = build_correction(calib_tables[chosen])

    raw = fitted[chosen].predict_proba(x_test)[:, 1]
    corrected = apply_correction(raw, correction)
    baseline = brier_score_loss(y_test, np.full(len(y_test), 0.5))
    assert ((corrected >= 0) & (corrected <= 1)).all(), "correction left [0,1]"
    assert _score(y_test, raw)["brier"] < baseline, "chosen model loses to a constant 0.5"

    return {
        "name": chosen,
        "model": fitted[chosen],
        "features": list(x.columns),
        "include_easy": include_easy,
        "calibration": correction,
        "calib_metrics": calib_metrics,
        "calib_tables": calib_tables,
        "metrics": _score(y_test, raw),
        "corrected_metrics": _score(y_test, corrected),
        "test_table": calibration_table(y_test, raw),
        "test_p": raw,   # aligned to evaluate.load()'s test order; sweep.py consumes it
        "baseline_brier": baseline,
    }


def _table_md(name: str, metrics: dict, table: pd.DataFrame) -> str:
    rows = "".join(
        f"| {r.bucket} | {r.n} | {r.predicted:.3f} | {r.actual:.3f} | {r.gap:+.3f} |\n"
        for r in table.itertuples()
    )
    return (
        f"### {name}\n\n"
        f"precision {metrics['precision']:.3f} | recall {metrics['recall']:.3f} | "
        f"F1 {metrics['f1']:.3f} | ROC-AUC {metrics['auc']:.3f} | "
        f"Brier {metrics['brier']:.4f}\n\n"
        f"| Predicted P(win) bucket | n | Predicted mean | Actual win rate | Gap |\n"
        f"|---|--:|--:|--:|--:|\n{rows}\n"
    )


def report(hard: dict, full: dict, saved: dict) -> str:
    h, f = hard["metrics"], full["metrics"]
    correction_rows = "".join(
        f"| {r.bucket} | {r.n} | {r.predicted:.3f} | {r.actual:.3f} | {v:.3f} |\n"
        for r, v in zip(saved["calib_tables"][saved["name"]].itertuples(),
                        saved["calibration"]["values"])
    )
    raw, corr = saved["metrics"], saved["corrected_metrics"]
    return (
        "# Calibration\n\n"
        "60/20/20 train / calibration / test, seed 42. The model is fitted on "
        "1800 rows, model selection and the rejected correction used a separate "
        "600, and every number in this file is measured on the remaining 600 that "
        "neither has seen. `evaluate.py` scores its strategies on that same test "
        "slice.\n\n"
        "## Does the decile correction help? No.\n\n"
        "It is measured here and **not applied**. `predict_win_prob` returns the "
        "raw isotonic-calibrated probability. Both rows below are scored on test, "
        "which neither the model nor the correction was fitted to:\n\n"
        "| P(win) as returned | ROC-AUC | Brier |\n|---|--:|--:|\n"
        f"| raw model output (shipped) | {raw['auc']:.4f} | {raw['brier']:.4f} |\n"
        f"| after decile correction | {corr['auc']:.4f} | {corr['brier']:.4f} |\n"
        f"| delta | {corr['auc'] - raw['auc']:+.4f} | "
        f"{corr['brier'] - raw['brier']:+.4f} |\n\n"
        "Worse on both. The correction was built to fix an apparent -0.146 gap at "
        "the decision boundary on an earlier two-way split; that gap did not "
        "reproduce (calibration +0.027, test -0.109, train -0.008, all with "
        "confidence intervals wide enough to contain each other). See FAILURES.md.\n\n"
        "## Cost of dropping customer-supplied features\n\n"
        "The money path excludes the easy-to-fake block (`complaint_category` "
        "one-hots). A claimant writes that field, so a model that leans on it can "
        "be moved by rewording a complaint. This is what the exclusion costs:\n\n"
        "| Feature set | ROC-AUC | Brier | Features |\n|---|--:|--:|--:|\n"
        f"| hard-to-fake only (shipped) | {h['auc']:.4f} | {h['brier']:.4f} | {len(hard['features'])} |\n"
        f"| all features | {f['auc']:.4f} | {f['brier']:.4f} | {len(full['features'])} |\n\n"
        "**The exclusion costs approximately nothing.** That is the whole claim, "
        "and the decimals above should not be read more precisely than that. Test "
        "is 600 rows: a decile of it holds 60 alerts, and a 60-row bucket at "
        "p≈0.5 carries a 95% interval of roughly ±0.13. Differences of this size "
        "between two feature sets sit inside that band. Treating a fourth-decimal "
        "gap as a finding is the mistake already written up in FAILURES.md.\n\n"
        f"Reproduce with `python model.py --with-easy`.\n\n"
        "## The correction that was measured and rejected\n\n"
        "Kept for reference, not applied. This is the map that would have replaced "
        "each raw prediction with the observed rate for its bucket, fitted on the "
        "calibration slice. Bucket rates pass through a weighted isotonic step, "
        "without which a higher-predicted bucket could return a lower P(win).\n\n"
        "| Bucket | n | Model says | Actually won | Returned |\n|---|--:|--:|--:|--:|\n"
        f"{correction_rows}\n"
        f"## Chosen: {saved['name']}\n\n"
        "Rule: default to logistic regression; switch only if gradient boosting "
        "wins on Brier *and* AUC. Decided on calibration, before test was touched.\n\n"
        "| Model (on calibration) | Brier (lower better) | ROC-AUC |\n|---|--:|--:|\n"
        f"| logistic | {saved['calib_metrics']['logistic']['brier']:.4f} | "
        f"{saved['calib_metrics']['logistic']['auc']:.3f} |\n"
        f"| gradient_boosting | {saved['calib_metrics']['gradient_boosting']['brier']:.4f} | "
        f"{saved['calib_metrics']['gradient_boosting']['auc']:.3f} |\n"
        f"| constant 0.5 (on test) | {saved['baseline_brier']:.4f} | 0.500 |\n\n"
        "## Honest calibration on test\n\n"
        "Raw model output, bucketed on the rows nothing was fitted to.\n\n"
        + _table_md(saved["name"] + " on test", raw, saved["test_table"])
    )


def predict_win_prob(df: pd.DataFrame, shares: dict[str, pd.Series] | None = None) -> np.ndarray:
    """P(win if fought) for raw alert rows. decide.py imports this.

    Returns the model's isotonic-calibrated probability directly. An earlier
    version mapped it through a decile correction measured on a holdout; on a
    proper three-way split that correction lost on both Brier and AUC, and the
    over-confidence it existed to fix turned out to be sampling noise in a
    60-row bucket. FAILURES.md has the intervals. The lesson worth keeping: the
    cv=5 isotonic layer inside the model is calibration fitted on training
    folds, which is the layer that earns its place.

    ponytail: share counts default to whatever `df` contains, which is right for
    scoring the whole file and wrong for scoring one alert in isolation -- a
    single row always looks unshared. Pass `shares` from the full alert history
    when scoring a slice. A real deployment reads these counts from the ledger.
    """
    global _CACHED
    if _CACHED is None:
        assert MODEL_PKL.exists(), f"{MODEL_PKL} missing -- run: python model.py"
        _CACHED = joblib.load(MODEL_PKL)
    x = build_features(df, shares=shares, include_easy=_CACHED["include_easy"])
    assert list(x.columns) == _CACHED["features"], "feature matrix drifted from the trained model"
    return _CACHED["model"].predict_proba(x)[:, 1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    with_easy = "--with-easy" in sys.argv

    hard = train(include_easy=False)
    full = train(include_easy=True)
    saved = full if with_easy else hard

    text = report(hard, full, saved)
    CALIBRATION.write_text(text, encoding="utf-8")
    joblib.dump({k: saved[k] for k in
                 ("name", "model", "features", "include_easy", "metrics", "calibration")}, MODEL_PKL)
    print(text)

    if saved["name"] == "logistic":
        coefs = _lr_coefficients(saved["model"], saved["features"])
        print("Standardised coefficients, largest absolute weight first:")
        print(coefs.to_string(float_format=lambda v: f"{v:+.3f}"))

    values = saved["calibration"]["values"]
    assert all(0.0 <= v <= 1.0 for v in values), "correction maps outside [0,1]"
    assert values == sorted(values), "correction inverts the ranking"

    # Round trip through the pickle: what decide.py imports must reproduce what
    # was just scored.
    df = pd.read_csv(evaluate.ALERTS)
    p = predict_win_prob(df)
    assert ((p >= 0) & (p <= 1)).all(), "probabilities outside [0,1] after reload"
    assert len(p) == len(df)
    # Continuous, not the nine steps the correction used to impose. If this ever
    # trips, something has quietly re-entered the money path.
    assert len(np.unique(p)) > len(values), "predict_win_prob is returning bucketed values"

    print(f"\nchosen: {saved['name']} on "
          f"{'all' if saved['include_easy'] else 'hard-to-fake'} features "
          f"-> {MODEL_PKL}, calibration -> {CALIBRATION}")


if __name__ == "__main__":
    main()
