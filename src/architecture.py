"""
Arquitectura del modelo: ResNet50 preentrenada en ImageNet + Transfer Learning.

Pipeline:  Imagen -> ResNet50 (conv base) -> GlobalAveragePooling2D -> Dropout -> Dense(1, sigmoid)

Diseñado para clasificación BINARIA (NORMAL vs PNEUMONIA) en esta v1, con `num_classes`
como parámetro para ampliar a multiclase (`build_model(num_classes=3)` cambia la salida a
softmax y la loss a `categorical_crossentropy`).

Entrenamiento en dos fases (transfer learning "de libro"):
  Fase A: `build_model(base_trainable=False)` -> solo se entrena la cabeza nueva.
  Fase B: `unfreeze_for_finetuning(modelo_fase_A, fine_tune_at=143)` -> se descongelan las
          últimas capas de ResNet50 y se sigue entrenando EL MISMO modelo con un LR bajo.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.resnet50 import ResNet50

from src.config import IMG_CHANNELS, IMG_SIZE, SEED

BASE_MODEL_NAME = "resnet50"


def set_seeds(seed: int = SEED) -> None:
    """Fija semillas para reproducibilidad razonable (no 100% determinista en GPU,
    pero suficiente para que las comparaciones entre runs de MLflow sean justas)."""
    tf.keras.utils.set_random_seed(seed)


def build_model(
    dropout: float = 0.3,
    num_classes: int = 1,
    base_trainable: bool = False,
    fine_tune_at: int = None,
    weights: str = "imagenet",
) -> tf.keras.Model:
    """Construye el modelo de clasificación.

    Args:
        dropout: tasa de Dropout antes de la capa densa final.
        num_classes: 1 para binario (sigmoid), >1 para multiclase (softmax).
        base_trainable: si False, TODA la base de ResNet50 queda congelada (Fase A).
        fine_tune_at: si `base_trainable=True`, índice de capa a partir del cual se
            descongela la base. Las capas anteriores a este índice permanecen congeladas
            para no destruir los features genéricos aprendidos en ImageNet.
        weights: "imagenet" (por defecto) o `None` para inicializar la base al azar --
            útil solo para pruebas rápidas del pipeline sin internet, nunca para el
            modelo final (perdería toda la ventaja del transfer learning).
    """
    base_model = ResNet50(
        weights=weights,
        include_top=False,
        input_shape=(*IMG_SIZE, IMG_CHANNELS),
    )
    _set_trainable(base_model, base_trainable, fine_tune_at)

    inputs = layers.Input(shape=(*IMG_SIZE, IMG_CHANNELS))
    # training=False: las BatchNormalization de la base siguen en modo inferencia incluso
    # al descongelar capas (si no, sus estadísticas se reajustan con lotes de 32 imágenes
    # de rayos X y el fine-tuning se degrada).
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="prediction")(x)

    return models.Model(inputs, outputs, name="aixray_resnet50")


def _set_trainable(base_model: tf.keras.Model, trainable: bool, fine_tune_at: int = None) -> None:
    base_model.trainable = trainable
    if trainable and fine_tune_at is not None:
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False


def unfreeze_for_finetuning(model: tf.keras.Model, fine_tune_at: int = 143) -> tf.keras.Model:
    """Fase B sobre un modelo YA entrenado en Fase A: descongela la base desde
    `fine_tune_at` conservando los pesos de la cabeza. Hay que volver a compilar después
    (con un learning rate bajo) para que el cambio de `trainable` tenga efecto."""
    _set_trainable(model.get_layer(BASE_MODEL_NAME), True, fine_tune_at)
    return model


def count_trainable_params(model: tf.keras.Model) -> int:
    return int(sum(tf.keras.backend.count_params(w) for w in model.trainable_weights))


def compile_model(model: tf.keras.Model, learning_rate: float, num_classes: int = 1) -> None:
    loss = "binary_crossentropy" if num_classes == 1 else "categorical_crossentropy"
    metrics = ["accuracy", tf.keras.metrics.AUC(name="auc")]
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=metrics,
    )
