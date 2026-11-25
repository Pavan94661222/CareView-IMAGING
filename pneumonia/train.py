"""Training / evaluation CLI for the pneumonia classifier.

Example
-------
::

    python -m pneumonia.train --epochs 8 --fine-tune-epochs 4 --batch-size 32

The script

1. downloads the Kaggle dataset through KaggleHub and discovers its layout,
2. reports the class distribution and rebalances it (class weights or
   controlled undersampling),
3. builds a stratified validation split *from the training folder* so the
   original ``test`` split stays untouched,
4. trains a transfer-learning model (frozen head, then fine-tuning),
5. picks the decision threshold on the validation set,
6. evaluates on the untouched test set and stores the model + metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from . import config
from .data import (
    build_dataset,
    cap_per_class,
    class_counts,
    compute_class_weights,
    dataset_summary,
    download_dataset,
    index_split,
    stratified_split,
    undersample,
    verify_records,
)
from .inference import default_metadata, save_metadata, tta_views, utc_now
from .metrics import (
    compute_metrics,
    find_best_threshold,
    format_report,
    plot_confusion_matrix,
    plot_curves,
    plot_threshold_sweep,
    plot_training_history,
    save_metrics,
)
from .model import (
    SUPPORTED_BACKBONES,
    build_model,
    count_parameters,
    normalize_backbone,
    unfreeze_for_fine_tuning,
)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pneumonia.train",
        description="Train the chest X-ray pneumonia classifier.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--backbone", default=config.BACKBONE, choices=sorted(set(SUPPORTED_BACKBONES)))
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE[0])
    parser.add_argument("--epochs", type=int, default=8, help="Frozen-backbone epochs.")
    parser.add_argument("--fine-tune-epochs", type=int, default=4, help="Fine-tuning epochs (0 disables).")
    parser.add_argument("--fine-tune-ratio", type=float, default=0.25, help="Fraction of the backbone unfrozen.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--dense-units", type=int, default=128)
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Stratified validation fraction taken from the train split.")
    parser.add_argument("--val-source", choices=["split", "original"], default="split",
                        help="'split' builds a stratified validation set from train/; "
                             "'original' uses the tiny val/ folder shipped with the dataset.")
    parser.add_argument("--imbalance", choices=["class_weight", "undersample", "none"],
                        default="class_weight", help="Strategy used to counter the 1:2.9 imbalance.")
    parser.add_argument("--majority-ratio", type=float, default=1.0,
                        help="Only for --imbalance undersample: majority/minority ratio to keep.")
    parser.add_argument("--max-train-per-class", type=int, default=config.MAX_TRAIN_PER_CLASS,
                        help="Keep at most this many training images per class (0 = all).")
    parser.add_argument("--max-val-per-class", type=int, default=config.MAX_VAL_PER_CLASS,
                        help="Keep at most this many validation images per class (0 = all).")
    parser.add_argument("--max-test-per-class", type=int, default=config.MAX_TEST_PER_CLASS,
                        help="Keep at most this many test images per class (0 = all).")
    parser.add_argument("--no-augment", action="store_true", help="Disable the augmentation block.")
    parser.add_argument("--no-cache", action="store_true", help="Disable in-memory dataset caching.")
    parser.add_argument("--no-verify", action="store_true", help="Skip the corrupt-image pre-scan.")
    parser.add_argument("--min-recall", type=float, default=None,
                        help="Constrain the threshold search, e.g. 0.95 for a screening setting.")
    parser.add_argument("--threshold-metric", choices=["accuracy", "balanced_accuracy", "f1"],
                        default="accuracy",
                        help="Validation objective used to select the classification threshold.")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--model-path", type=Path, default=config.DEFAULT_MODEL_PATH)
    parser.add_argument("--metadata-path", type=Path, default=config.DEFAULT_METADATA_PATH)
    parser.add_argument("--reports-dir", type=Path, default=config.REPORTS_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="Skip the download and use an already extracted dataset.")
    return parser


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def collect_probabilities(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    tta: bool = config.TTA_ENABLED,
    n_views: int = config.TTA_VIEWS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (y_true, y_prob) for a dataset that yields ``(image, label[, weight])``.

    When *tta* is true every image is scored on several deterministic views and
    the probabilities are averaged - the exact same procedure the backend uses.
    """
    truths: List[np.ndarray] = []
    probs: List[np.ndarray] = []
    for batch in dataset:
        if isinstance(batch, (tuple, list)):
            images, labels = batch[0], batch[1]
        else:  # pragma: no cover - defensive
            raise ValueError("Unexpected dataset element structure.")

        images = np.asarray(images, dtype=np.float32)
        if tta:
            views = tta_views(images, n_views)
            total = None
            for view in views:
                predicted = model(view, training=False).numpy().reshape(-1)
                total = predicted if total is None else total + predicted
            predictions = total / float(len(views))
        else:
            predictions = model(images, training=False).numpy().reshape(-1)

        probs.append(predictions)
        truths.append(np.asarray(labels).reshape(-1))
    if not probs:
        raise RuntimeError("Dataset produced no batches.")
    return np.concatenate(truths), np.concatenate(probs)


def build_callbacks(checkpoint_path: Path, patience: int, monitor: str = "val_auc") -> List[tf.keras.callbacks.Callback]:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor=monitor,
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, patience // 2), min_lr=1e-7, verbose=1
        ),
    ]


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    set_global_seed(args.seed)

    img_size = (args.img_size, args.img_size)
    backbone = normalize_backbone(args.backbone)

    print("=" * 78)
    print(f"Pneumonia detection - training ({backbone}, {img_size[0]}x{img_size[0]})")
    print(f"TensorFlow {tf.__version__}")
    print("=" * 78)

    # ---- 1. dataset ---------------------------------------------------
    if args.dataset_dir:
        from .data import resolve_dataset_root

        root = resolve_dataset_root(Path(args.dataset_dir))
    else:
        root = download_dataset()
    print(f"[data] dataset root: {root}")

    train_records = index_split(root, "train")
    test_records = index_split(root, "test")
    print(f"[data] train split : {dataset_summary(train_records)}")
    print(f"[data] test  split : {dataset_summary(test_records)}")

    if not args.no_verify:
        train_records = verify_records(train_records, drop_corrupt=True)
        test_records = verify_records(test_records, drop_corrupt=True)
        print(f"[data] after verification -> train: {len(train_records)}, test: {len(test_records)}")

    if args.max_test_per_class > 0:
        before = dataset_summary(test_records)
        test_records = cap_per_class(test_records, args.max_test_per_class, args.seed)
        print(f"[cap] test  per class <= {args.max_test_per_class}: {before} -> {dataset_summary(test_records)}")

    # ---- 2. train / validation split (test stays untouched) -----------
    if args.val_source == "original" and (Path(root) / "val").is_dir():
        val_records = index_split(root, "val")
        if not args.no_verify:
            val_records = verify_records(val_records, drop_corrupt=True)
        print(f"[data] validation : {dataset_summary(val_records)}  (original val/ folder)")
    else:
        train_records, val_records = stratified_split(train_records, args.val_split, args.seed)
        print(f"[data] validation : {dataset_summary(val_records)}  (stratified from train/)")

    if args.max_val_per_class > 0:
        before = dataset_summary(val_records)
        val_records = cap_per_class(val_records, args.max_val_per_class, args.seed)
        print(f"[cap] val   per class <= {args.max_val_per_class}: {before} -> {dataset_summary(val_records)}")

    if args.max_train_per_class > 0:
        before = dataset_summary(train_records)
        train_records = cap_per_class(train_records, args.max_train_per_class, args.seed)
        print(f"[cap] train per class <= {args.max_train_per_class}: {before} -> {dataset_summary(train_records)}")

    if not val_records:
        raise RuntimeError("Validation set is empty - relax --max-val-per-class / --val-split.")

    # ---- 3. class balancing ------------------------------------------
    sample_weights: Optional[Dict[int, float]] = None
    if args.imbalance == "undersample":
        before = dataset_summary(train_records)
        train_records = undersample(train_records, args.seed, args.majority_ratio)
        print(f"[balance] undersampling: {before} -> {dataset_summary(train_records)}")
    elif args.imbalance == "class_weight":
        sample_weights = compute_class_weights(train_records)
        pretty = {config.CLASS_NAMES[k]: round(v, 4) for k, v in sorted(sample_weights.items())}
        print(f"[balance] class weights: {pretty}")
    else:
        print("[balance] none (raw distribution will be used)")
    print(f"[data] final train: {dataset_summary(train_records)}")

    # ---- 4. tf.data pipelines ----------------------------------------
    use_cache = not args.no_cache
    train_ds = build_dataset(
        train_records, img_size, args.batch_size, shuffle=True,
        seed=args.seed, cache=use_cache, drop_remainder=True, sample_weights=sample_weights,
    )
    val_ds = build_dataset(
        val_records, img_size, args.batch_size, shuffle=False, cache=use_cache,
    )
    test_ds = build_dataset(
        test_records, img_size, args.batch_size, shuffle=False, cache=use_cache,
    )

    # ---- 5. model -----------------------------------------------------
    model = build_model(
        backbone=backbone,
        img_size=img_size,
        dropout=args.dropout,
        dense_units=args.dense_units,
        learning_rate=args.learning_rate,
        augment=not args.no_augment,
    )
    params = count_parameters(model)
    print(f"[model] parameters: {params['trainable']:,} trainable / {params['total']:,} total")

    checkpoint_path = Path(args.model_path).with_suffix(".best.keras")
    callbacks = build_callbacks(checkpoint_path, patience=4)

    history: Dict[str, List[float]] = {}
    print("\n[phase 1] training the classification head (backbone frozen)")
    phase1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    history.update({key: list(values) for key, values in phase1.history.items()})

    if args.fine_tune_epochs > 0:
        print(f"\n[phase 2] fine-tuning the deepest {args.fine_tune_ratio:.0%} of {backbone}")
        trainable = unfreeze_for_fine_tuning(
            model, learning_rate=args.fine_tune_lr, unfreeze_ratio=args.fine_tune_ratio
        )
        print(f"[phase 2] {trainable} backbone layers unfrozen")
        phase2 = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=build_callbacks(checkpoint_path, patience=3),
            verbose=2,
        )
        for key, values in phase2.history.items():
            history.setdefault(key, []).extend(list(values))

    # Restore the best checkpoint of the whole run.
    if checkpoint_path.is_file():
        model = tf.keras.models.load_model(checkpoint_path)
        print(f"[model] restored best checkpoint from {checkpoint_path.name}")

    # ---- 6. threshold selection on validation -------------------------
    y_val, p_val = collect_probabilities(model, val_ds)
    threshold, val_metrics = find_best_threshold(
        y_val, p_val, metric=args.threshold_metric, min_recall=args.min_recall
    )
    print("\n" + format_report(
        val_metrics, f"Validation set (threshold chosen for {args.threshold_metric})"
    ))

    # A screening-oriented alternative is reported but never used silently.
    screening_threshold, screening_metrics = find_best_threshold(
        y_val, p_val, metric="f1", min_recall=0.97
    )
    print(
        f"[threshold] chosen={threshold:.2f} for {args.threshold_metric} | "
        "the 0.97-recall operating point would be "
        f"{screening_threshold:.2f} (recall={screening_metrics['recall']:.3f}, "
        f"specificity={screening_metrics['specificity']:.3f})"
    )

    # ---- 7. final evaluation on the untouched test set ----------------
    y_test, p_test = collect_probabilities(model, test_ds)
    test_metrics = compute_metrics(y_test, p_test, threshold)
    print("\n" + format_report(test_metrics, "Test set (never seen during training)"))

    # ---- 8. persist artifacts ----------------------------------------
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = {
        "training_history": plot_training_history(history, reports_dir / "training_history.png"),
        "confusion_matrix": plot_confusion_matrix(test_metrics, reports_dir / "confusion_matrix.png"),
        "roc_pr": plot_curves(y_test, p_test, test_metrics, reports_dir / "roc_pr_curves.png"),
        "threshold_sweep": plot_threshold_sweep(
            y_val, p_val, threshold, reports_dir / "threshold_sweep.png"
        ),
    }

    np.save(reports_dir / "test_probabilities.npy", p_test)
    np.save(reports_dir / "test_labels.npy", y_test)

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)

    metadata = default_metadata()
    metadata.update(
        {
            "model_name": f"pneumonia_{backbone}",
            "backbone": backbone,
            "img_size": [img_size[0], img_size[1]],
            "class_names": list(config.CLASS_NAMES),
            "positive_class": config.POSITIVE_CLASS,
            "threshold": float(threshold),
            "screening_threshold": float(screening_threshold),
            "created_at": utc_now(),
            "tensorflow_version": tf.__version__,
            "input_range": [0.0, 255.0],
            "input_color": "RGB",
            "parameters": params,
            "dataset": {
                "kaggle_dataset": config.KAGGLE_DATASET,
                "root": str(root),
                "train": class_counts(train_records),
                "validation": class_counts(val_records),
                "test": class_counts(test_records),
                "imbalance_strategy": args.imbalance,
                "class_weights": sample_weights,
                "val_source": args.val_source,
                "max_per_class": {
                    "train": args.max_train_per_class,
                    "val": args.max_val_per_class,
                    "test": args.max_test_per_class,
                },
            },
            "training": {
                "epochs": args.epochs,
                "fine_tune_epochs": args.fine_tune_epochs,
                "fine_tune_ratio": args.fine_tune_ratio,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "fine_tune_learning_rate": args.fine_tune_lr,
                "dropout": args.dropout,
                "augmentation": not args.no_augment,
                "seed": args.seed,
                "epochs_completed": len(history.get("loss", [])),
                "threshold_metric": args.threshold_metric,
            },
            "tta": bool(config.TTA_ENABLED),
            "tta_views": int(config.TTA_VIEWS),
            "validation_metrics": val_metrics,
            "test_metrics": test_metrics,
            "history": history,
            "reports": {k: (str(v) if v else None) for k, v in plot_paths.items()},
        }
    )
    save_metadata(metadata, Path(args.metadata_path))
    save_metrics(test_metrics, reports_dir / "test_metrics.json", extra={
        "validation_metrics": val_metrics,
        "metadata": {k: v for k, v in metadata.items() if k not in {"history", "test_metrics", "validation_metrics"}},
    })

    print("\n[save] model    ->", model_path)
    print("[save] metadata ->", args.metadata_path)
    print("[save] report   ->", reports_dir / "test_metrics.json")
    if checkpoint_path.is_file():
        checkpoint_path.unlink()

    print("\nDone. Start the API with:  uvicorn backend.main:app --reload --port 8000")
    return 0


if __name__ == "__main__":
    sys.exit(main())
