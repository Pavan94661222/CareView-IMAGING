"""Offline evaluation of a saved model artifact.

::

    python -m pneumonia.evaluate --split test
    python -m pneumonia.evaluate --split test --threshold 0.4 --min-recall 0.95

Prints accuracy / precision / recall / F1 / specificity / ROC-AUC and the
confusion matrix, and writes the plots next to the other reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf

from . import config
from .data import (
    cap_per_class,
    class_counts,
    dataset_summary,
    download_dataset,
    index_split,
    max_per_class_for_split,
    verify_records,
    resolve_dataset_root,
)
from .metrics import (
    compute_metrics,
    find_best_threshold,
    format_report,
    plot_confusion_matrix,
    plot_curves,
    plot_threshold_sweep,
    save_metrics,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pneumonia.evaluate",
        description="Evaluate the saved pneumonia classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--model-path", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=config.DEFAULT_METADATA_PATH)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-verify", action="store_true")
    parser.add_argument("--max-per-class", type=int, default=None,
                        help="Override the per-class cap for this split "
                             "(default: the configured value for the split, 0 = no cap).")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override the threshold stored in the metadata.")
    parser.add_argument("--no-tta", action="store_true",
                        help="Disable test-time augmentation for this run.")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Also report the best threshold achieving this recall.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    from .data import build_dataset
    from .inference import PneumoniaPredictor

    predictor = PneumoniaPredictor(args.model_path, args.metadata_path)
    if not predictor.is_ready:
        print(f"[evaluate] ERROR: {predictor.health()['error']}")
        return 1

    root = resolve_dataset_root(Path(args.dataset_dir)) if args.dataset_dir else download_dataset()
    records = index_split(root, args.split)
    if not args.no_verify:
        records = verify_records(records, drop_corrupt=True)

    cap = args.max_per_class if args.max_per_class is not None else max_per_class_for_split(args.split)
    if cap > 0:
        before = dataset_summary(records)
        records = cap_per_class(records, cap, config.RANDOM_SEED)
        print(f"[evaluate] per-class cap {cap}: {before} -> {dataset_summary(records)}")

    print(f"[evaluate] model      : {args.model_path}")
    print(f"[evaluate] backbone   : {predictor.backbone} @ {predictor.img_size}")
    print(f"[evaluate] TTA        : {'off' if args.no_tta else f'{predictor.n_tta_views} views'}")
    print(f"[evaluate] split      : {args.split} ({dataset_summary(records)})")

    dataset = build_dataset(
        records, tuple(predictor.img_size), args.batch_size, shuffle=False, cache=False
    )

    truths, probabilities = [], []
    for images, labels in dataset:
        probabilities.append(predictor.probabilities(images.numpy(), tta=not args.no_tta))
        truths.append(np.asarray(labels).reshape(-1))
    y_true = np.concatenate(truths)
    y_prob = np.concatenate(probabilities)

    threshold = args.threshold if args.threshold is not None else predictor.threshold
    metrics = compute_metrics(y_true, y_prob, threshold)
    print("\n" + format_report(metrics, f"{args.split} split @ threshold {threshold:.2f}"))

    if args.min_recall is not None:
        alt_threshold, alt_metrics = find_best_threshold(
            y_true, y_prob, metric="f1", min_recall=args.min_recall
        )
        print("\n" + format_report(
            alt_metrics, f"{args.split} split @ recall>={args.min_recall:.2f} (threshold {alt_threshold:.2f})"
        ))
        metrics = alt_metrics

    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion_matrix(metrics, reports_dir / f"confusion_matrix_{args.split}.png")
    plot_curves(y_true, y_prob, metrics, reports_dir / f"roc_pr_curves_{args.split}.png")
    plot_threshold_sweep(y_true, y_prob, metrics["threshold"], reports_dir / f"threshold_sweep_{args.split}.png")
    save_metrics(
        metrics,
        reports_dir / f"{args.split}_metrics.json",
        extra={
            "split": args.split,
            "class_distribution": class_counts(records),
            "model": predictor.info(),
            "tensorflow_version": tf.__version__,
        },
    )
    print(f"\n[evaluate] report written to {reports_dir / f'{args.split}_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
