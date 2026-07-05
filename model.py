"""Train and evaluate no-show prediction models.

Trains an interpretable logistic regression (the model used for per-appointment
explanations) and a random forest (for a performance comparison). Reports the
metrics that matter for an imbalanced problem — ROC-AUC, PR-AUC, and recall on
the no-show class — not accuracy. Persists everything the dashboard needs.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ingest.config import (
    CATEGORICAL_FEATURES,
    MODEL_PATH,
    MODEL_SEED,
    NUMERIC_FEATURES,
    SCORED_PATH,
    TARGET,
    TEST_SIZE,
)

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ]
    )


def _make_pipeline(estimator) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("clf", estimator)])


def _class_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    pred = (prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "recall": float(recall_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion": confusion_matrix(y_true, pred).tolist(),
    }


def _gains(y_true: np.ndarray, prob: np.ndarray) -> dict:
    """Cumulative share of no-shows captured as you target the riskiest first."""
    order = np.argsort(-prob)
    y_sorted = np.asarray(y_true)[order]
    cum_captured = np.cumsum(y_sorted) / max(y_sorted.sum(), 1)
    frac_pop = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    top_decile_capture = float(cum_captured[max(int(0.1 * len(y_sorted)) - 1, 0)])
    return {
        "frac_pop": frac_pop.tolist(),
        "cum_captured": cum_captured.tolist(),
        "top_decile_capture": top_decile_capture,
    }


def train(df: pd.DataFrame) -> dict:
    X = df[FEATURES]
    y = df[TARGET].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=MODEL_SEED, stratify=y
    )

    lr = _make_pipeline(
        LogisticRegression(class_weight="balanced", max_iter=1000)
    ).fit(X_tr, y_tr)
    rf = _make_pipeline(
        RandomForestClassifier(
            n_estimators=250, class_weight="balanced",
            max_depth=12, random_state=MODEL_SEED, n_jobs=-1,
        )
    ).fit(X_tr, y_tr)

    prob_lr = lr.predict_proba(X_te)[:, 1]
    prob_rf = rf.predict_proba(X_te)[:, 1]

    metrics = {"logistic_regression": _class_metrics(y_te, prob_lr),
               "random_forest": _class_metrics(y_te, prob_rf)}

    # feature names + LR coefficients (for global + per-appointment explanations)
    feat_names = lr.named_steps["prep"].get_feature_names_out().tolist()
    lr_coef = lr.named_steps["clf"].coef_[0].tolist()
    rf_import = rf.named_steps["clf"].feature_importances_.tolist()

    bundle = {
        "lr_pipeline": lr,
        "rf_pipeline": rf,
        "feature_names": feat_names,
        "lr_coef": lr_coef,
        "rf_importances": rf_import,
        "metrics": metrics,
        "gains_lr": _gains(y_te.to_numpy(), prob_lr),
        "y_test": y_te.to_numpy().tolist(),
        "prob_lr": prob_lr.tolist(),
        "prob_rf": prob_rf.tolist(),
        "features": FEATURES,
    }
    joblib.dump(bundle, MODEL_PATH)

    # score every appointment with the interpretable model for the dashboard tables
    scored = df.copy()
    scored["no_show_prob"] = lr.predict_proba(X)[:, 1]
    scored.to_parquet(SCORED_PATH, index=False)

    print(f"[model] saved model -> {MODEL_PATH}")
    print(f"[model] LR  ROC-AUC {metrics['logistic_regression']['roc_auc']:.3f} "
          f"recall {metrics['logistic_regression']['recall']:.3f}")
    print(f"[model] RF  ROC-AUC {metrics['random_forest']['roc_auc']:.3f} "
          f"recall {metrics['random_forest']['recall']:.3f}")
    print(f"[model] targeting riskiest 10% captures "
          f"{bundle['gains_lr']['top_decile_capture']:.1%} of no-shows")
    return bundle


def explain_appointment(bundle: dict, row: pd.DataFrame, top_n: int = 6) -> list[tuple[str, float]]:
    """Per-appointment explanation: contribution = LR coefficient x transformed value."""
    lr = bundle["lr_pipeline"]
    transformed = lr.named_steps["prep"].transform(row[bundle["features"]])[0]
    coef = np.asarray(bundle["lr_coef"])
    contributions = coef * transformed
    names = bundle["feature_names"]
    pairs = sorted(zip(names, contributions), key=lambda p: abs(p[1]), reverse=True)
    return [(n, float(v)) for n, v in pairs[:top_n]]


if __name__ == "__main__":
    from features import build_features
    from ingest.load import load_appointments

    train(build_features(load_appointments()))
