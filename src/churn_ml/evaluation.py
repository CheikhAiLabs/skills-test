from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from churn_ml.config import PipelineConfig
from churn_ml.exceptions import ModelValidationError


def compute_binary_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "average_precision": round(float(average_precision_score(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, predictions, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, predictions)), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y_true, predictions)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "predicted_positive_rate": round(float(predictions.mean()), 4),
        "threshold": round(float(threshold), 4),
    }


def select_operating_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    min_precision: float,
    min_recall: float,
) -> tuple[float, dict[str, float]]:
    candidate_thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_metrics: dict[str, float] | None = None
    fallback_metrics: dict[str, float] | None = None

    for threshold in candidate_thresholds:
        metrics = compute_binary_metrics(y_true, probabilities, float(threshold))
        if fallback_metrics is None or metrics["f1"] > fallback_metrics["f1"]:
            fallback_metrics = metrics
        if metrics["precision"] >= min_precision and metrics["recall"] >= min_recall:
            if best_metrics is None or metrics["f1"] > best_metrics["f1"]:
                best_threshold = float(threshold)
                best_metrics = metrics

    if best_metrics is not None:
        return best_threshold, best_metrics
    if fallback_metrics is None:
        raise ModelValidationError("Unable to evaluate threshold candidates.")
    return float(fallback_metrics["threshold"]), fallback_metrics


def evaluate_baseline(
    X_train: Any,
    y_train: np.ndarray,
    X_test: Any,
    y_test: np.ndarray,
) -> dict[str, float]:
    baseline = DummyClassifier(strategy="prior")
    baseline.fit(X_train, y_train)
    baseline_probabilities = baseline.predict_proba(X_test)[:, 1]
    return compute_binary_metrics(y_test, baseline_probabilities, 0.5)


def enforce_model_gates(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    config: PipelineConfig,
) -> None:
    failures: list[str] = []
    if metrics["roc_auc"] < config.metrics.min_roc_auc:
        failures.append(
            f"roc_auc {metrics['roc_auc']:.4f} < {config.metrics.min_roc_auc:.4f}"
        )
    if metrics["average_precision"] < config.metrics.min_average_precision:
        failures.append(
            "average_precision "
            f"{metrics['average_precision']:.4f} < {config.metrics.min_average_precision:.4f}"
        )
    if metrics["f1"] < config.metrics.min_f1:
        failures.append(f"f1 {metrics['f1']:.4f} < {config.metrics.min_f1:.4f}")
    if metrics["recall"] < config.metrics.min_recall:
        failures.append(
            f"recall {metrics['recall']:.4f} < {config.metrics.min_recall:.4f}"
        )
    if metrics["roc_auc"] <= baseline_metrics["roc_auc"]:
        failures.append(
            "model roc_auc did not beat baseline "
            f"({metrics['roc_auc']:.4f} <= {baseline_metrics['roc_auc']:.4f})"
        )

    if failures:
        raise ModelValidationError("Model promotion failed: " + "; ".join(failures))
