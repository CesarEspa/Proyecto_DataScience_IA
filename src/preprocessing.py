"""
Pipeline de `tf.data` para cargar, decodificar, preprocesar y (opcionalmente) aumentar
las imágenes de rayos X, a partir del manifiesto construido en `src/dataset.py`.

Punto importante: ResNet50 con pesos de ImageNet espera entradas 224x224x3 preprocesadas
con `tensorflow.keras.applications.resnet50.preprocess_input` (que centra los canales
según las estadísticas de ImageNet), NO una normalización genérica a [0, 1].

Mejora respecto a la versión de clase: `preprocess_image_bytes()` es la MISMA función
de decodificación + resize + preprocess que usa el entrenamiento, y es la que llama la
API. Antes la API redimensionaba con PIL (bicúbico) y el entrenamiento con
`tf.image.resize` (bilineal): un pequeño desfase entrenamiento/inferencia
(*training-serving skew*) que ahora no existe.
"""

import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input

from src.config import BATCH_SIZE, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE


def build_augmentation(horizontal_flip: bool = True) -> tf.keras.Sequential:
    """Capas de augmentation (solo para train).

    `horizontal_flip=False` existe para la Variante C: en una radiografía de tórax el
    flip horizontal produce una anatomía que no existe en la vida real (corazón a la
    derecha, *situs inversus* es rarísimo), así que es discutible como augmentation.
    """
    layers = []
    if horizontal_flip:
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=SEED))
    layers += [
        tf.keras.layers.RandomRotation(0.05, seed=SEED),
        tf.keras.layers.RandomZoom(0.1, seed=SEED),
        tf.keras.layers.RandomContrast(0.1, seed=SEED),
    ]
    return tf.keras.Sequential(layers, name="augmentation")


def _decode_and_resize(raw: tf.Tensor) -> tf.Tensor:
    # decode_image soporta JPEG y PNG; expand_animations=False garantiza un tensor 3D.
    img = tf.io.decode_image(raw, channels=3, expand_animations=False)
    # antialias=True: al reducir una radiografía de ~2000 px a 224 px evita aliasing.
    img = tf.image.resize(img, IMG_SIZE, antialias=True)  # bilineal, float32
    return img


def preprocess_image_bytes(image_bytes: bytes) -> tf.Tensor:
    """Bytes de una imagen (JPEG/PNG) -> tensor (1, 224, 224, 3) listo para el modelo.
    Usado por la API: exactamente el mismo preprocesamiento que val/test."""
    img = _decode_and_resize(tf.constant(image_bytes))
    img = preprocess_input(img)
    return tf.expand_dims(img, axis=0)


def build_dataset(
    paths,
    labels,
    split: str,
    batch_size: int = BATCH_SIZE,
    shuffle_buffer: int = 1024,
    horizontal_flip: bool = True,
) -> tf.data.Dataset:
    """Construye un `tf.data.Dataset` listo para `model.fit` / `model.evaluate`.

    - `split="train"`: aplica shuffle + data augmentation.
    - `split in ("val", "test")`: sin augmentation ni shuffle (evaluación determinista).
    """
    paths = tf.constant(list(paths))
    labels = tf.constant(list(labels), dtype=tf.float32)
    augmentation = build_augmentation(horizontal_flip) if split == "train" else None

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if split == "train":
        ds = ds.shuffle(shuffle_buffer, seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, label):
        img = _decode_and_resize(tf.io.read_file(path))
        if augmentation is not None:
            img = augmentation(img, training=True)
        img = preprocess_input(img)
        return img, label

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def dataset_from_manifest(
    manifest_df, split: str, batch_size: int = BATCH_SIZE, horizontal_flip: bool = True
) -> tf.data.Dataset:
    """Atajo: filtra el manifiesto por la columna `split` y arma el `tf.data.Dataset`."""
    subset = manifest_df[manifest_df["split"] == split]
    return build_dataset(
        subset["path"].tolist(), subset["label"].tolist(), split, batch_size,
        horizontal_flip=horizontal_flip,
    )
