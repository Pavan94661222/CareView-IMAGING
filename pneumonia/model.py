"""Keras model factory (transfer learning) for NORMAL / PNEUMONIA classification.

The returned model consumes images in the canonical **[0, 255] float32** format
produced by :mod:`pneumonia.preprocessing`.  Every backbone specific
normalisation is baked into the graph, therefore the saved artifact is fully
self-contained and the backend cannot accidentally use a different pipeline.

Data augmentation layers are part of the model as well.  Keras activates them
only when the model is called with ``training=True`` (i.e. during ``fit``), so
inference is deterministic.
"""

from __future__ import annotations

from typing import Dict, Tuple

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
from tensorflow.keras.applications import (
    DenseNet121,
    EfficientNetB0,
    EfficientNetB3,
    EfficientNetV2B0,
    EfficientNetV2B1,
    MobileNetV2,
    ResNet50V2,
    VGG16,
    densenet,
    efficientnet,
    mobilenet_v2,
    resnet_v2,
    vgg16,
)

from . import config

#: ``preprocess`` is applied *inside* the model to map [0, 255] RGB to whatever
#: the backbone expects.  ``None`` means the backbone handles it internally
#: (EfficientNet / EfficientNetV2 ship with their own ``Rescaling`` +
#: ``Normalization`` layers).
BACKBONES: Dict[str, Dict] = {
    "efficientnetv2b0": {
        "factory": EfficientNetV2B0,
        "preprocess": None,
        "min_size": 32,
        "description": "EfficientNetV2B0 (ImageNet) - stronger and faster to converge than B0",
    },
    "efficientnetv2b1": {
        "factory": EfficientNetV2B1,
        "preprocess": None,
        "min_size": 32,
        "description": "EfficientNetV2B1 (ImageNet) - higher capacity than V2B0",
    },
    "efficientnetb0": {
        "factory": EfficientNetB0,
        "preprocess": None,
        "min_size": 32,
        "description": "EfficientNetB0 (ImageNet) - best accuracy/size trade-off",
    },
    "efficientnetb3": {
        "factory": EfficientNetB3,
        "preprocess": None,
        "min_size": 32,
        "description": "EfficientNetB3 (ImageNet) - larger, native 300px input",
    },
    "densenet121": {
        "factory": DenseNet121,
        "preprocess": densenet.preprocess_input,
        "min_size": 32,
        "description": "DenseNet121 (ImageNet) - dense skip connections, strong on radiographs",
    },
    "resnet50v2": {
        "factory": ResNet50V2,
        "preprocess": resnet_v2.preprocess_input,
        "min_size": 32,
        "description": "ResNet50V2 (ImageNet) - high capacity residual network",
    },
    "mobilenetv2": {
        "factory": MobileNetV2,
        "preprocess": mobilenet_v2.preprocess_input,
        "min_size": 32,
        "description": "MobileNetV2 (ImageNet) - fastest, smallest",
    },
    "vgg16": {
        "factory": VGG16,
        "preprocess": vgg16.preprocess_input,
        "min_size": 48,
        "description": "VGG16 (ImageNet) - heavy, high capacity",
    },
    # ``efficientnet`` is an alias used by some scripts.
    "efficientnet": {
        "factory": EfficientNetB0,
        "preprocess": None,
        "min_size": 32,
        "description": "Alias of EfficientNetB0",
    },
}

SUPPORTED_BACKBONES = tuple(BACKBONES)

#: backbones whose normalisation is *not* embedded in the pretrained graph.
_EXTERNAL_PREPROCESS = {
    "mobilenetv2": mobilenet_v2.preprocess_input,
    "vgg16": vgg16.preprocess_input,
    "densenet121": densenet.preprocess_input,
    "resnet50v2": resnet_v2.preprocess_input,
}


@tf.keras.utils.register_keras_serializable(package="pneumonia")
class BackbonePreprocess(layers.Layer):
    """Applies a backbone's ``preprocess_input`` inside the model graph.

    Implemented as a registered layer (instead of ``Lambda``) so that the saved
    ``.keras`` artifact reloads in a fresh process without ``custom_objects``.
    """

    def __init__(self, backbone: str = "mobilenetv2", **kwargs):
        super().__init__(**kwargs)
        key = backbone.lower()
        if key not in _EXTERNAL_PREPROCESS:
            raise ValueError(f"{backbone} has no external preprocessing.")
        self.backbone = key

    def call(self, inputs):
        return _EXTERNAL_PREPROCESS[self.backbone](inputs)

    def get_config(self):
        config = super().get_config()
        config["backbone"] = self.backbone
        return config


def normalize_backbone(name: str) -> str:
    key = (name or "").strip().lower()
    if key not in BACKBONES:
        raise ValueError(
            f"Unsupported backbone '{name}'. Choose one of {', '.join(sorted(BACKBONES))}."
        )
    return "efficientnetb0" if key == "efficientnet" else key


def build_augmentation(img_size: Tuple[int, int]) -> tf.keras.Sequential:
    """Gentle augmentation suitable for chest radiographs.

    Horizontal flipping is included: it is a standard, well validated
    augmentation for chest radiograph *classification* (the left/right
    distinction matters for lateralised findings, not for the presence of a
    consolidation). The same flip is used by test-time augmentation, so the
    training and inference distributions stay aligned.
    """
    return tf.keras.Sequential(
        [
            layers.RandomFlip(mode="horizontal"),
            layers.RandomRotation(factor=0.04, fill_mode="nearest"),
            layers.RandomZoom(height_factor=0.10, width_factor=0.10, fill_mode="nearest"),
            layers.RandomTranslation(
                height_factor=0.06, width_factor=0.06, fill_mode="nearest"
            ),
            layers.RandomContrast(factor=0.12),
        ],
        name="augmentation",
    )


def build_model(
    backbone: str = config.BACKBONE,
    img_size: Tuple[int, int] = config.IMG_SIZE,
    dropout: float = 0.3,
    dense_units: int = 128,
    learning_rate: float = 1e-3,
    augment: bool = True,
    trainable_base: bool = False,
    weights: str = "imagenet",
    l2: float = 1e-4,
) -> tf.keras.Model:
    """Create and compile the classifier.

    Parameters
    ----------
    backbone
        One of :data:`SUPPORTED_BACKBONES`.
    augment
        When ``True`` the augmentation block is embedded in the model and only
        active during training.
    trainable_base
        When ``True`` the backbone weights are unfrozen immediately (used for
        the fine-tuning phase).
    """
    key = normalize_backbone(backbone)
    spec = BACKBONES[key]
    if img_size[0] < spec["min_size"] or img_size[1] < spec["min_size"]:
        raise ValueError(f"{key} requires an input of at least {spec['min_size']}px.")

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3), name="xray")
    x = inputs

    if augment:
        x = build_augmentation(img_size)(x)

    if spec["preprocess"] is not None:
        x = BackbonePreprocess(key, name=f"{key}_preprocess")(x)

    base = spec["factory"](
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights=weights,
    )
    base.trainable = trainable_base

    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Dropout(dropout, name="head_dropout")(x)
    if dense_units > 0:
        x = layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=regularizers.l2(l2),
            name="head_dense",
        )(x)
        x = layers.Dropout(dropout / 2, name="head_dropout_2")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="pneumonia")(x)

    model = models.Model(inputs, outputs, name=f"pneumonia_{key}")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def get_backbone_module(model: tf.keras.Model) -> tf.keras.Model:
    """Locate the nested transfer-learning backbone inside *model*."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name != "augmentation":
            return layer
    raise ValueError("No nested backbone model found.")


def unfreeze_for_fine_tuning(
    model: tf.keras.Model,
    learning_rate: float = 1e-5,
    unfreeze_ratio: float = 0.25,
    keep_batch_norm_frozen: bool = True,
) -> int:
    """Make the deepest ``unfreeze_ratio`` of the backbone trainable.

    Returns the number of backbone layers that became trainable.
    """
    backbone = get_backbone_module(model)
    backbone.trainable = True

    total = len(backbone.layers)
    first_trainable = int(total * (1.0 - unfreeze_ratio))

    trainable_count = 0
    for index, layer in enumerate(backbone.layers):
        is_bn = isinstance(layer, layers.BatchNormalization)
        if index < first_trainable:
            layer.trainable = False
        elif is_bn and keep_batch_norm_frozen:
            # Keep running statistics of BN layers frozen: with a small medical
            # dataset re-estimating them destroys the ImageNet features.
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return trainable_count


def count_parameters(model: tf.keras.Model) -> Dict[str, int]:
    trainable = int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))
    non_trainable = int(
        sum(tf.keras.backend.count_params(w) for w in model.non_trainable_weights)
    )
    return {"trainable": trainable, "non_trainable": non_trainable, "total": trainable + non_trainable}


def layer_summary(model: tf.keras.Model) -> str:
    lines = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


__all__ = [
    "BACKBONES",
    "SUPPORTED_BACKBONES",
    "build_model",
    "build_augmentation",
    "normalize_backbone",
    "unfreeze_for_fine_tuning",
    "count_parameters",
    "layer_summary",
]
