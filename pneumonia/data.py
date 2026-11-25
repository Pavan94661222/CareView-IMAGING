"""Dataset download, discovery, balancing and ``tf.data`` pipeline.

The Kaggle *Chest X-Ray Images (Pneumonia)* dataset is downloaded with
``kagglehub`` and its real on-disk hierarchy is discovered automatically::

    chest_xray/
        train/{NORMAL,PNEUMONIA}
        test/{NORMAL,PNEUMONIA}
        val/{NORMAL,PNEUMONIA}

Nothing is hard-coded: the folder that actually contains ``train/`` and
``test/`` is located at runtime, so a kagglehub cache path, a manually
extracted archive or a copy in ``data/`` all work.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from . import config
from .config import CLASS_NAMES, IMAGE_EXTENSIONS
from .preprocessing import resize_canonical

AUTOTUNE = tf.data.AUTOTUNE


@dataclass(frozen=True)
class Record:
    """A single labelled image on disk."""

    path: str
    label: int

    @property
    def class_name(self) -> str:
        return CLASS_NAMES[self.label]


# --------------------------------------------------------------------------
# Download / discovery
# --------------------------------------------------------------------------
def download_dataset(force: bool = False) -> Path:
    """Download the dataset through KaggleHub and return the split root folder.

    Parameters
    ----------
    force
        Re-run :func:`kagglehub.dataset_download` instead of reusing a local
        ``PNEUMONIA_DATASET_DIR``.
    """
    if config.LOCAL_DATASET_DIR and not force:
        local = Path(config.LOCAL_DATASET_DIR)
        if local.exists():
            return resolve_dataset_root(local)

    import kagglehub

    # Official KaggleHub download (documented entry point for this dataset).
    downloaded = kagglehub.dataset_download(config.KAGGLE_DATASET)
    print(f"[data] KaggleHub dataset path: {downloaded}")
    return resolve_dataset_root(Path(downloaded))


def _looks_like_root(path: Path) -> bool:
    return (path / "train").is_dir() and (path / "test").is_dir()


def resolve_dataset_root(base: Path, max_depth: int = 3) -> Path:
    """Find the directory that holds ``train/`` and ``test/`` below *base*."""
    base = Path(base).expanduser()
    if not base.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {base}")
    if _looks_like_root(base):
        return base

    frontier = [(base, 0)]
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for child in sorted(p for p in current.iterdir() if p.is_dir()):
            if _looks_like_root(child):
                return child
            frontier.append((child, depth + 1))

    raise FileNotFoundError(
        f"Could not locate a 'train'/'test' hierarchy inside {base}. "
        "Extract the chest_xray dataset and try again."
    )


def available_splits(root: Path) -> List[str]:
    return [split for split in config.SPLITS if (Path(root) / split).is_dir()]


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------
def label_from_dirname(name: str) -> Optional[int]:
    upper = name.upper()
    if "NORMAL" in upper:
        return 0
    if "PNEUMONIA" in upper:
        return 1
    return None


def index_split(root: Path, split: str) -> List[Record]:
    """Return every labelled image of *split* using the real folder names."""
    split_dir = Path(root) / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split '{split}' not found in {root}")

    records: List[Record] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = label_from_dirname(class_dir.name)
        if label is None:
            continue  # ignore stray folders such as __MACOSX
        for file in sorted(class_dir.rglob("*")):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                records.append(Record(str(file), label))

    if not records:
        raise RuntimeError(
            f"No images found in {split_dir}. Expected sub-folders "
            f"{CLASS_NAMES} filled with {IMAGE_EXTENSIONS} files."
        )
    return records


def class_counts(records: Iterable[Record]) -> Dict[str, int]:
    counter = Counter(rec.label for rec in records)
    return {name: counter.get(idx, 0) for idx, name in enumerate(CLASS_NAMES)}


def dataset_summary(records: Iterable[Record]) -> str:
    counts = class_counts(records)
    total = sum(counts.values())
    if total == 0:
        return "no images"
    parts = [f"{name}={count} ({count / total:.1%})" for name, count in counts.items()]
    return f"total={total} | " + " | ".join(parts)


def verify_records(
    records: Sequence[Record], drop_corrupt: bool = True, verbose: bool = True
) -> List[Record]:
    """Drop files that Pillow cannot open (a few truncated files ship with the dataset)."""
    from PIL import Image

    good: List[Record] = []
    broken: List[str] = []
    for rec in records:
        try:
            with Image.open(rec.path) as img:
                img.verify()
            good.append(rec)
        except Exception:
            broken.append(rec.path)

    if broken and verbose:
        print(f"[data] WARNING: {len(broken)} unreadable image(s) detected:")
        for path in broken[:5]:
            print(f"        - {path}")
        if len(broken) > 5:
            print(f"        ... and {len(broken) - 5} more")
    if not drop_corrupt:
        return list(records)
    return good


# --------------------------------------------------------------------------
# Balancing
# --------------------------------------------------------------------------
def compute_class_weights(records: Iterable[Record]) -> Dict[int, float]:
    """Balanced class weights (``n_samples / (n_classes * n_samples_j)``)."""
    labels = [rec.label for rec in records]
    counter = Counter(labels)
    total = len(labels)
    n_classes = len(counter)
    if total == 0 or n_classes == 0:
        return {}
    return {int(cls): total / (n_classes * count) for cls, count in counter.items()}


def undersample(
    records: Sequence[Record],
    seed: int = config.RANDOM_SEED,
    majority_ratio: float = 1.0,
) -> List[Record]:
    """Randomly down-sample the majority class instead of duplicating images.

    ``majority_ratio=1.0`` produces a perfectly balanced set.  Unlike
    over-sampling this never repeats an image, so the model cannot memorise one.
    """
    grouped: Dict[int, List[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.label, []).append(rec)

    if len(grouped) < 2:
        return list(records)

    smallest = min(len(items) for items in grouped.values())
    target = max(int(round(smallest * majority_ratio)), smallest)
    rng = random.Random(seed)

    balanced: List[Record] = []
    for label, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        balanced.extend(items[: min(target, len(items))])
    rng.shuffle(balanced)
    return balanced


def cap_per_class(
    records: Sequence[Record],
    limit: int,
    seed: int = config.RANDOM_SEED,
) -> List[Record]:
    """Keep at most *limit* randomly chosen images per class.

    ``limit <= 0`` disables the cap.  Images are selected, never duplicated, so
    the class balance requested by the caller is reached without repeating a
    single file.
    """
    if limit <= 0:
        return list(records)

    grouped: Dict[int, List[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.label, []).append(rec)

    rng = random.Random(seed)
    capped: List[Record] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        capped.extend(items[:limit])
    rng.shuffle(capped)
    return capped


def max_per_class_for_split(split: str) -> int:
    """Configured per-class cap for ``train`` / ``val`` / ``test`` (0 = no cap)."""
    return {
        "train": config.MAX_TRAIN_PER_CLASS,
        "val": config.MAX_VAL_PER_CLASS,
        "test": config.MAX_TEST_PER_CLASS,
    }.get(split, 0)


def stratified_split(
    records: Sequence[Record],
    val_ratio: float = 0.15,
    seed: int = config.RANDOM_SEED,
) -> Tuple[List[Record], List[Record]]:
    """Split *records* into (train, val) keeping the class ratio identical.

    The original ``test`` split is never touched by this function, which keeps
    the final evaluation unbiased.
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1.")

    grouped: Dict[int, List[Record]] = {}
    for rec in records:
        grouped.setdefault(rec.label, []).append(rec)
    if len(grouped) < 2:
        raise ValueError("Stratified split needs at least two classes.")

    rng = random.Random(seed)
    train: List[Record] = []
    val: List[Record] = []
    for label in sorted(grouped):
        items = list(grouped[label])
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_ratio)))
        val.extend(items[:n_val])
        train.extend(items[n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


# --------------------------------------------------------------------------
# tf.data pipeline
# --------------------------------------------------------------------------
def _decode_record(path: tf.Tensor, label: tf.Tensor, img_size: Tuple[int, int]):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    return resize_canonical(image, img_size), label


def build_dataset(
    records: Sequence[Record],
    img_size: Tuple[int, int] = config.IMG_SIZE,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = config.RANDOM_SEED,
    cache: bool = False,
    drop_remainder: bool = False,
    sample_weights: Optional[Dict[int, float]] = None,
) -> tf.data.Dataset:
    """Build a batched ``tf.data`` pipeline yielding ``(float32[0,255], label)``.

    When *sample_weights* is given the batches become ``(image, label, weight)``
    tuples, which is how Keras receives class weights for a dataset input.
    """
    if not records:
        raise ValueError("Cannot build a dataset from an empty record list.")

    paths = [rec.path for rec in records]
    labels = np.asarray([rec.label for rec in records], dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(
        lambda path, label: _decode_record(path, label, img_size),
        num_parallel_calls=AUTOTUNE,
        deterministic=not shuffle,
    )
    if cache:
        dataset = dataset.cache()

    if sample_weights:
        weight_values = tf.constant(
            [float(sample_weights.get(i, 1.0)) for i in range(len(CLASS_NAMES))],
            dtype=tf.float32,
        )
        dataset = dataset.map(
            lambda image, label: (
                image,
                label,
                tf.gather(weight_values, tf.cast(label, tf.int32)),
            ),
            num_parallel_calls=AUTOTUNE,
        )

    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset.prefetch(AUTOTUNE)
