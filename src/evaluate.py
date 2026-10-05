"""Reusable validation metrics for probability scores and ranked targeting."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def top_k_metrics(y_true, probabilities, fractions=(0.05, 0.10, 0.20)):
    """Rank by probability; select ceil(k * n), breaking ties by input order."""
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities)
    total = len(labels)
    total_responders = int(labels.sum())
    overall_rate = total_responders / total
    ranking = np.argsort(-scores, kind="stable")
    results = {}
    for fraction in fractions:
        count = math.ceil(fraction * total)
        captured = int(labels[ranking[:count]].sum())
        selected_rate = captured / count
        results[f"top_{fraction * 100:g}_pct"] = {
            "selected_count": count,
            "responders_captured": captured,
            "responder_capture_rate": captured / total_responders if total_responders else 0.0,
            "response_rate": selected_rate,
            "lift": selected_rate / overall_rate if overall_rate else None,
        }
    return results


def probability_summary(probabilities):
    """Summarize validation scores without fitting or choosing a threshold."""
    scores = np.asarray(probabilities)
    return {
        "min": float(np.min(scores)),
        "p01": float(np.quantile(scores, 0.01)),
        "p25": float(np.quantile(scores, 0.25)),
        "median": float(np.median(scores)),
        "mean": float(np.mean(scores)),
        "p75": float(np.quantile(scores, 0.75)),
        "p99": float(np.quantile(scores, 0.99)),
        "max": float(np.max(scores)),
    }


def evaluate_predictions(y_true, probabilities, threshold=0.5):
    """Score validation predictions; threshold metrics are reporting references."""
    labels = np.asarray(y_true)
    scores = np.asarray(probabilities)
    predictions = (scores >= threshold).astype(int)
    return {
        "validation_count": len(labels),
        "responders": int(labels.sum()),
        "overall_response_rate": float(labels.mean()),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "reporting_threshold": threshold,
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "probability_summary": probability_summary(scores),
        "ranking": top_k_metrics(labels, scores),
    }
