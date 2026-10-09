"""
Configuración central del proyecto AI-XRay.

Todas las rutas y constantes que otros módulos (notebooks, src/, api/) necesitan
viven aquí para no repetirlas ni desincronizarlas entre archivos.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

# Raíz del proyecto AI-XRay/ (src/config.py -> AI-XRay/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# El dataset "Pediatric Chest X-ray Pneumonia" (Kaggle, andrewmvd) se espera
# descomprimido un nivel arriba de AI-XRay/:
#   Clase 4/
#   ├── archive (2)/Pediatric Chest X-ray Pneumonia/{train,test}/{NORMAL,PNEUMONIA}
#   └── AI-XRay/                                          <- PROJECT_ROOT
# Se puede sobreescribir con la variable de entorno AIXRAY_DATA_DIR sin tocar código.
DEFAULT_DATA_DIR = PROJECT_ROOT.parent / "archive (2)" / "Pediatric Chest X-ray Pneumonia"
DATA_DIR = Path(os.environ.get("AIXRAY_DATA_DIR", str(DEFAULT_DATA_DIR)))

# Carpetas de salida (se crean solas la primera vez que se escribe en ellas).
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"  # tablas/figuras finales que alimentan el Executive Summary
MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest.csv"
SPLIT_MANIFEST_PATH = PROJECT_ROOT / "data" / "manifest_split.csv"

# Tracking de MLflow: backend SQLite local + artefactos SIEMPRE en AI-XRay/mlruns
# (no relativos a la carpeta desde donde se ejecuta el notebook -- en la versión
# original los artefactos terminaban en notebooks/mlruns/).
MLFLOW_TRACKING_URI = os.environ.get(
    "AIXRAY_MLFLOW_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_ARTIFACT_ROOT = (PROJECT_ROOT / "mlruns").as_uri()
MLFLOW_EXPERIMENT_NAME = "AI-XRay"
MLFLOW_MODEL_NAME = "aixray_pneumonia_classifier"
MLFLOW_MODEL_ALIAS = "staging"  # alias moderno de MLflow 3 (equivalente al stage "Staging")

# --------------------------------------------------------------------------
# Constantes del problema
# --------------------------------------------------------------------------

CLASSES = ["NORMAL", "PNEUMONIA"]  # índice 0 = NORMAL, índice 1 = PNEUMONIA (v1 binaria)
IMG_SIZE = (224, 224)  # tamaño esperado por ResNet50 con pesos de ImageNet
IMG_CHANNELS = 3
BATCH_SIZE = 32

# Subconjunto balanceado para el laboratorio (dataset completo = 5,856 imágenes).
DEMO_SUBSET_TOTAL = int(os.environ.get("AIXRAY_SUBSET_TOTAL", 2000))
DEMO_SUBSET_PER_CLASS = DEMO_SUBSET_TOTAL // len(CLASSES)  # 1000 NORMAL + 1000 PNEUMONIA

# Split propio (NO se usa el split original train/test del dataset, ver data/README.md).
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Umbral de decisión por defecto y objetivo clínico de sensibilidad para la variante
# "umbral clínico" (ver notebooks/03_training.ipynb, Variante B).
DEFAULT_THRESHOLD = 0.5
TARGET_RECALL = 0.95

SEED = 42


def setup_mlflow():
    """Configura tracking URI + experimento con la carpeta de artefactos fija en
    AI-XRay/mlruns. Devuelve el experimento. Usar SIEMPRE esto en vez de
    `mlflow.set_experiment(...)` directo, para que todos los notebooks escriban en el
    mismo sitio sin importar desde dónde se ejecuten."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_ROOT)
    return mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
