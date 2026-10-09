"""
Carga del modelo entrenado y lógica de inferencia, separada de los endpoints de FastAPI
(`main.py`): el modelo se carga UNA sola vez al iniciar el proceso, no en cada petición.

Configuración por variables de entorno (antes estaba fija en el código y las variables
que mencionaba el README nunca se leían):

- `AIXRAY_MODEL_SOURCE`: "local" (por defecto) o "mlflow".
    - "local": carga `models/aixray_model.keras` (lo exporta `04_evaluation.ipynb`).
    - "mlflow": carga desde el Model Registry por alias (`models:/<nombre>@staging`).
- `AIXRAY_MODEL_PATH`: ruta alternativa al `.keras` (modo local).
- `AIXRAY_THRESHOLD`: umbral de decisión. Si no se define, se usa el umbral clínico guardado
  en `models/model_card.json` por `04_evaluation.ipynb`; si tampoco existe, 0.5.
"""

import json
import os
from pathlib import Path

import numpy as np

from src.config import (
    CLASSES,
    DEFAULT_THRESHOLD,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
)

MODEL_CARD_PATH = MODELS_DIR / "model_card.json"


class ModelNotAvailableError(RuntimeError):
    """El modelo no se pudo cargar (archivo inexistente, registry vacío...)."""


class InvalidImageError(ValueError):
    """Los bytes recibidos no son una imagen JPEG/PNG decodificable."""


class ModelService:
    """Envoltorio simple: carga el modelo una vez y expone `predict_image(bytes)`."""

    def __init__(self):
        self.source = os.environ.get("AIXRAY_MODEL_SOURCE", "local").lower()
        self.local_path = Path(os.environ.get("AIXRAY_MODEL_PATH", MODELS_DIR / "aixray_model.keras"))
        self.model = None
        self.card = {}
        self.threshold = DEFAULT_THRESHOLD
        self.version_label = "sin cargar"

    def load(self):
        if self.model is not None:
            return
        if self.source == "local":
            if not self.local_path.exists():
                raise ModelNotAvailableError(
                    f"No se encontró el modelo en {self.local_path}. Corre primero "
                    "notebooks/03_training.ipynb y 04_evaluation.ipynb, o define AIXRAY_MODEL_PATH."
                )
            from tensorflow.keras.models import load_model

            self.model = load_model(self.local_path, compile=False)
            if MODEL_CARD_PATH.exists():
                self.card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
            self.version_label = (
                f"{MLFLOW_MODEL_NAME} v{self.card['registry_version']} (local)"
                if "registry_version" in self.card else f"local:{self.local_path.name}"
            )
        elif self.source == "mlflow":
            import mlflow
            import mlflow.keras

            mlflow.set_tracking_uri(os.environ.get("AIXRAY_MLFLOW_URI", MLFLOW_TRACKING_URI))
            try:
                mv = mlflow.MlflowClient().get_model_version_by_alias(MLFLOW_MODEL_NAME, MLFLOW_MODEL_ALIAS)
                self.model = mlflow.keras.load_model(f"models:/{MLFLOW_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}")
            except Exception as e:  # registry vacío, alias inexistente, artefactos movidos...
                raise ModelNotAvailableError(f"No se pudo cargar el modelo desde MLflow: {e}") from e
            self.card = {k: v for k, v in (mv.tags or {}).items()}
            self.version_label = f"{MLFLOW_MODEL_NAME} v{mv.version} (@{MLFLOW_MODEL_ALIAS})"
        else:
            raise ModelNotAvailableError(f"AIXRAY_MODEL_SOURCE inválido: {self.source!r} (usa local o mlflow)")

        env_thr = os.environ.get("AIXRAY_THRESHOLD")
        self.threshold = float(env_thr if env_thr is not None else self.card.get("threshold_clinical", DEFAULT_THRESHOLD))

    def is_loaded(self) -> bool:
        return self.model is not None

    def predict_image(self, image_bytes: bytes) -> dict:
        if self.model is None:
            self.load()
        import tensorflow as tf

        from src.preprocessing import preprocess_image_bytes  # MISMO preprocesamiento que el entrenamiento

        try:
            x = preprocess_image_bytes(image_bytes)
        except (tf.errors.InvalidArgumentError, ValueError) as e:
            raise InvalidImageError("El archivo no es una imagen JPEG/PNG válida.") from e

        # predict_on_batch usa el grafo compilado (llamar al modelo en modo eager consume
        # mucha más RAM en CPU)
        prob_pneumonia = float(np.asarray(self.model.predict_on_batch(x)).ravel()[0])
        label = CLASSES[1] if prob_pneumonia >= self.threshold else CLASSES[0]
        return {
            "prediction": label,
            "probability": round(prob_pneumonia, 4),
            "threshold": round(self.threshold, 4),
            "model_version": self.version_label,
        }


# Instancia única compartida por toda la app (patrón singleton simple).
model_service = ModelService()
