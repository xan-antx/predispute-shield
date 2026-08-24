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
    """Turn the decile table into a lookup: raw probability -> observed rate.

    The measured gap is worst exactly where money gets committed. On the held-out
    set the top decile claimed 0.935 and delivered 0.867, and the 0.879-0.912
    bucket claimed 0.900 and delivered 0.856. An EV rule sends precisely those
    alerts to a fight, so shipping the raw number means paying for six or seven
    points of confidence that were never there.

    The raw bucket rates are not monotone -- small-sample noise puts a lower
    observed rate on a higher-predicted bucket -- so they go through a weighted
    isotonic pass first. Without it, an alert with better evidence could come
    back with a lower P(win), which is indefensible in front of a merchant.

    ponytail: this is fitted on the same holdout the model is scored on, so any
    metric computed after correction is in-sample and not an honest estimate.
    It is applied anyway because being systematically conservative about money
    beats being statistically pure about a number nobody spends. The upgrade is
    a third split used only for calibration, which 3000 rows does not support.
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
    """Fit both candidates on one feature set, score on the holdout, pick one."""
    train_df, test_df = evaluate.load()
    assert (len(train_df), len(test_df)) == (2100, 900), "split does not match evaluate.py"
    assert not train_df.index.intersection(test_df.index).size, "train and test overlap"

    full = pd.concat([train_df, test_df]).sort_index()
    x = build_features(full, shares=share_counts(full), include_easy=include_easy)
    x_train, x_test = x.loc[train_df.index], x.loc[test_df.index]
    y_train, y_test = train_df[LABEL], test_df[LABEL]

    fitted, metrics, tables, probs = {}, {}, {}, {}
    for name, estimator in _models().items():
        estimator.fit(x_train, y_train)
        p = estimator.predict_proba(x_test)[:, 1]
        fitted[name], probs[name] = estimator, p
        metrics[name] = _score(y_test, p)
        tables[name] = calibration_table(y_test, p)

    lr, gb = metrics["logistic"], metrics["gradient_boosting"]
    # Interpretability is the tiebreak, not the metric. A coefficient I can read
    # out to a panel is worth more than a marginal lift I would have to defend
    # with a partial dependence plot.
    gb_wins = gb["brier"] < lr["brier"] and gb["auc"] > lr["auc"]
    chosen = "gradient_boosting" if gb_wins else "logistic"

    for name, p in probs.items():
        assert ((p >= 0) & (p <= 1)).all(), f"{name} produced a probability outside [0,1]"
    baseline = brier_score_loss(y_test, np.full(len(y_test), 0.5))
    assert metrics[chosen]["brier"] < baseline, "chosen model loses to a constant 0.5"

    correction = build_correction(tables[chosen])
    corrected = apply_correction(probs[chosen], correction)

    return {
        "name": chosen,
        "model": fitted[chosen],
        "features": list(x.columns),
        "include_easy": include_easy,
        "metrics": metrics[chosen],
        "all_metrics": metrics,
        "tables": tables,
        "calibration": correction,
        "baseline_brier": baseline,
        # In-sample by construction; kept to show the correction does what it claims.
        "corrected_metrics": _score(y_test, corrected),
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
        for r, v in zip(saved["tables"][saved["name"]].itertuples(), saved["calibration"]["values"])
    )
    return (
        "# Calibration\n\n"
        "900 held-out alerts, deciles of predicted P(win). Split and rows are "
        "identical to `evaluate.py`.\n\n"
        "## Cost of dropping customer-supplied features\n\n"
        "The money path excludes the easy-to-fake block (`complaint_category` "
        "one-hots). A claimant writes that field, so a model that leans on it can "
        "be moved by rewording a complaint. This is what the exclusion costs:\n\n"
        "| Feature set | ROC-AUC | Brier | Features |\n|---|--:|--:|--:|\n"
        f"| hard-to-fake only (shipped) | {h['auc']:.4f} | {h['brier']:.4f} | {len(hard['features'])} |\n"
        f"| all features | {f['auc']:.4f} | {f['brier']:.4f} | {len(full['features'])} |\n"
        f"| **delta** | **{h['auc'] - f['auc']:+.4f}** | **{h['brier'] - f['brier']:+.4f}** | "
        f"**{len(hard['features']) - len(full['features'])}** |\n\n"
        f"Reproduce with `python model.py --with-easy`.\n\n"
        "## Applied correction\n\n"
        "Raw predictions are mapped through the observed rate for their bucket "
        "before being returned by `predict_win_prob`. Bucket rates pass through a "
        "weighted isotonic step first, so the mapping cannot invert the ranking.\n\n"
        "| Bucket | n | Model says | Actually won | Returned |\n|---|--:|--:|--:|--:|\n"
        f"{correction_rows}\n"
        f"## Chosen: {saved['name']}\n\n"
        "Rule: default to logistic regression; switch only if gradient boosting "
        "wins on Brier *and* AUC.\n\n"
        "| Model | Brier (lower better) | ROC-AUC |\n|---|--:|--:|\n"
        f"| logistic | {saved['all_metrics']['logistic']['brier']:.4f} | "
        f"{saved['all_metrics']['logistic']['auc']:.3f} |\n"
        f"| gradient_boosting | {saved['all_metrics']['gradient_boosting']['brier']:.4f} | "
        f"{saved['all_metrics']['gradient_boosting']['auc']:.3f} |\n"
        f"| constant 0.5 | {saved['baseline_brier']:.4f} | 0.500 |\n\n"
        "## Per-model detail\n\n"
        + "".join(_table_md(n, saved["all_metrics"][n], saved["tables"][n]) for n in saved["tables"])
    )


def predict_win_prob(df: pd.DataFrame, shares: dict[str, pd.Series] | None = None) -> np.ndarray:
    """P(win if fought) for raw alert rows. decide.py imports this.

    The returned number is the *observed* win rate for the bucket the model's
    raw prediction falls into, not the model's own claim. On the holdout the top
    decile claimed 0.935 and delivered 0.867; an EV rule fed the raw number
    would authorise fights priced on six points of confidence that do not exist.
    See build_correction for how the mapping is fitted and what it costs.

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
    raw = _CACHED["model"].predict_proba(x)[:, 1]
    return apply_correction(raw, _CACHED["calibration"])


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
    # was just scored, correction included.
    df = pd.read_csv(evaluate.ALERTS)
    p = predict_win_prob(df)
    assert ((p >= 0) & (p <= 1)).all(), "probabilities outside [0,1] after reload"
    assert len(p) == len(df)
    assert set(np.unique(p)).issubset(set(values)), "correction not applied on reload"

    print(f"\nchosen: {saved['name']} on "
          f"{'all' if saved['include_easy'] else 'hard-to-fake'} features "
          f"-> {MODEL_PKL}, calibration -> {CALIBRATION}")


if __name__ == "__main__":
    main()
