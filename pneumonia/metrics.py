"""Evaluation metrics, threshold selection and report plots.

All numbers produced here are computed from real model outputs - nothing is
pre-filled or estimated.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from .config import CLASS_NAMES

MATPLOTLIB_AVAILABLE = True
try:  # pragma: no cover - environment dependent
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _ = plt
except Exception:  # pragma: no cover
    MATPLOTLIB_AVAILABLE = False


# --------------------------------------------------------------------------
# Metric computation
# --------------------------------------------------------------------------
def binarize(probabilities: Iterable[float], threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(probabilities, dtype=float).ravel() >= threshold).astype(int)


def compute_metrics(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict:
    """Full metric bundle for the positive (PNEUMONIA) class."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError("y_true and y_prob must have the same length.")

    y_pred = binarize(y_prob, threshold)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    has_both_classes = len(np.unique(y_true)) > 1

    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    npv = float(tn / (tn + fn)) if (tn + fn) else 0.0

    return {
        "threshold": float(threshold),
        "n_samples": int(y_true.size),
        "n_normal": int((y_true == 0).sum()),
        "n_pneumonia": int((y_true == 1).sum()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": specificity,
        "negative_predictive_value": npv,
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both_classes else None,
        "average_precision": (
            float(average_precision_score(y_true, y_prob)) if has_both_classes else None
        ),
        "confusion_matrix": {
            "labels": list(CLASS_NAMES),
            "matrix": matrix.tolist(),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }


def find_best_threshold(
    y_true: Sequence[int],
    y_prob: Sequence[float],
    metric: str = "f1",
    min_recall: Optional[float] = None,
    grid: Optional[np.ndarray] = None,
) -> Tuple[float, Dict]:
    """Pick the decision threshold that maximises *metric* on validation data.

    Parameters
    ----------
    min_recall
        Optional hard constraint, e.g. ``0.95`` for a screening tool where
        missing a pneumonia case is far worse than a false alarm.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if grid is None:
        grid = np.round(np.arange(0.05, 0.96, 0.01), 4)

    best_threshold = 0.5
    best_score = -np.inf
    best_metrics: Dict = {}

    valid_thresholds: List[float] = []
    for threshold in grid:
        metrics = compute_metrics(y_true, y_prob, float(threshold))
        if min_recall is not None and metrics["recall"] < min_recall:
            continue
        valid_thresholds.append(float(threshold))
        if metric == "f1":
            score = metrics["f1"]
        elif metric == "balanced_accuracy":
            score = metrics["balanced_accuracy"]
        elif metric == "specificity":
            score = metrics["specificity"]
        else:
            score = metrics[metric]

        # Accuracy ties prefer specificity; other objectives retain recall priority.
        better = score > best_score + 1e-9
        if abs(score - best_score) <= 1e-9 and metrics:
            if metric == "accuracy":
                specificity = metrics["specificity"]
                best_specificity = best_metrics.get("specificity", 0.0)
                better = specificity > best_specificity + 1e-9
                if abs(specificity - best_specificity) <= 1e-9:
                    better = metrics["recall"] > best_metrics.get("recall", 0.0)
            else:
                better = metrics["recall"] > best_metrics.get("recall", 0.0)
        if better:
            best_score = score
            best_threshold = float(threshold)
            best_metrics = metrics

    if not best_metrics:
        best_threshold = 0.5
        best_metrics = compute_metrics(y_true, y_prob, best_threshold)

    best_metrics["threshold_search"] = {
        "objective": metric,
        "min_recall": min_recall,
        "candidates": len(valid_thresholds),
    }
    return float(best_threshold), best_metrics


def format_report(metrics: Dict, title: str = "Evaluation") -> str:
    cm = metrics["confusion_matrix"]
    lines = [
        f"===== {title} =====",
        f"samples      : {metrics['n_samples']}  "
        f"(NORMAL={metrics['n_normal']}, PNEUMONIA={metrics['n_pneumonia']})",
        f"threshold    : {metrics['threshold']:.2f}",
        f"accuracy     : {metrics['accuracy']:.4f}",
        f"precision    : {metrics['precision']:.4f}  (PNEUMONIA)",
        f"recall       : {metrics['recall']:.4f}  (PNEUMONIA)",
        f"f1-score     : {metrics['f1']:.4f}  (PNEUMONIA)",
        f"specificity  : {metrics['specificity']:.4f}",
        f"balanced acc : {metrics['balanced_accuracy']:.4f}",
    ]
    if metrics.get("roc_auc") is not None:
        lines.append(f"roc-auc      : {metrics['roc_auc']:.4f}")
    if metrics.get("average_precision") is not None:
        lines.append(f"avg precision: {metrics['average_precision']:.4f}")
    lines += [
        "confusion matrix (rows=actual, cols=predicted):",
        "                NORMAL  PNEUMONIA",
        f"     NORMAL     {cm['tn']:>6}  {cm['fp']:>9}",
        f"  PNEUMONIA     {cm['fn']:>6}  {cm['tp']:>9}",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_training_history(history: Dict, output_path: Path) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (keys, title) in zip(
        axes,
        [
            (("accuracy", "val_accuracy"), "Accuracy"),
            (("loss", "val_loss"), "Loss"),
            (("recall", "val_recall"), "PNEUMONIA recall"),
        ],
    ):
        train_key, val_key = keys
        if train_key in history:
            ax.plot(history[train_key], label="train")
        if val_key in history:
            ax.plot(history[val_key], label="validation")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_confusion_matrix(metrics: Dict, output_path: Path) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    matrix = np.asarray(metrics["confusion_matrix"]["matrix"])
    fig, ax = plt.subplots(figsize=(5.5, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], list(CLASS_NAMES))
    ax.set_yticks([0, 1], list(CLASS_NAMES))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix (threshold={metrics['threshold']:.2f})")
    for row in range(2):
        for col in range(2):
            ax.text(
                col,
                row,
                str(matrix[row, col]),
                ha="center",
                va="center",
                color="black" if matrix[row, col] < matrix.max() / 2 else "white",
                fontsize=14,
            )
    fig.colorbar(image, ax=ax, fraction=0.046)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_curves(
    y_true: Sequence[int], y_prob: Sequence[float], metrics: Dict, output_path: Path
) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    if metrics.get("roc_auc") is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
        axes[0].plot([0, 1], [0, 1], "--", lw=1, color="grey")
        axes[0].set_xlabel("False positive rate")
        axes[0].set_ylabel("True positive rate")
        axes[0].set_title("ROC curve (PNEUMONIA)")
        axes[0].legend(loc="lower right")
        axes[0].grid(alpha=0.3)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(recall, precision, lw=2, color="darkorange")
    if metrics.get("average_precision") is not None:
        axes[1].set_title(f"Precision-Recall (AP = {metrics['average_precision']:.3f})")
    else:
        axes[1].set_title("Precision-Recall")
    axes[1].set_xlabel("Recall (PNEUMONIA)")
    axes[1].set_ylabel("Precision (PNEUMONIA)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


def plot_threshold_sweep(
    y_true: Sequence[int], y_prob: Sequence[float], best_threshold: float, output_path: Path
) -> Optional[Path]:
    if not MATPLOTLIB_AVAILABLE:
        return None
    grid = np.round(np.arange(0.05, 0.96, 0.01), 4)
    rows = [compute_metrics(y_true, y_prob, float(t)) for t in grid]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for key, label in (
        ("recall", "Recall (PNEUMONIA)"),
        ("specificity", "Specificity"),
        ("precision", "Precision"),
        ("f1", "F1"),
        ("accuracy", "Accuracy"),
    ):
        ax.plot(grid, [row[key] for row in rows], label=label, lw=1.8)
    ax.axvline(best_threshold, color="black", ls="--", lw=1.2)
    ax.text(best_threshold + 0.01, 0.03, f"chosen = {best_threshold:.2f}", fontsize=9)
    ax.set_xlabel("Decision threshold on P(PNEUMONIA)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax.set_title("Metric sensitivity to the decision threshold")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path


# --------------------------------------------------------------------------
# Persistence helpers
# --------------------------------------------------------------------------
def save_metrics(metrics: Dict, output_path: Path, extra: Optional[Dict] = None) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(metrics)
    if extra:
        payload.update(extra)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return output_path


def load_metrics(path: Path) -> Dict:
    with open(Path(path), "r", encoding="utf-8") as handle:
        return json.load(handle)
