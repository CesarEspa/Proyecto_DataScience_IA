"""
Pruebas de la API de AI-XRay.

    pytest tests/ -v

Las pruebas de contrato usan un modelo Keras DIMINUTO (misma entrada 224x224x3, salida
sigmoide) creado en un directorio temporal, así corren en segundos y sin depender de que
el modelo real ya esté entrenado. La última prueba usa el modelo real exportado si existe.
"""

import importlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _image_bytes(fmt: str = "PNG", size=(300, 260)) -> bytes:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 255, size=(size[1], size[0]), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format=fmt)
    return buf.getvalue()


def _make_client(monkeypatch, model_path: Path, threshold: str = None):
    """Recarga los módulos de la API con las variables de entorno indicadas."""
    monkeypatch.setenv("AIXRAY_MODEL_SOURCE", "local")
    monkeypatch.setenv("AIXRAY_MODEL_PATH", str(model_path))
    if threshold is not None:
        monkeypatch.setenv("AIXRAY_THRESHOLD", threshold)
    else:
        monkeypatch.delenv("AIXRAY_THRESHOLD", raising=False)
    import api.model_service
    import api.main
    importlib.reload(api.model_service)
    importlib.reload(api.main)
    from fastapi.testclient import TestClient
    return TestClient(api.main.app)


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    path = tmp_path_factory.mktemp("model") / "tiny.keras"
    model.save(path)
    return path


@pytest.fixture
def client(monkeypatch, tiny_model_path):
    with _make_client(monkeypatch, tiny_model_path, threshold="0.5") as c:
        yield c


# --------------------------------------------------------------------------- contrato

def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["threshold"] == 0.5


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg")])
def test_predict_valid_image(client, fmt, mime):
    r = client.post("/predict", files={"file": (f"rx.{fmt.lower()}", _image_bytes(fmt), mime)})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["prediction"] in {"NORMAL", "PNEUMONIA"}
    assert 0.0 <= body["probability"] <= 1.0
    assert body["prediction"] == ("PNEUMONIA" if body["probability"] >= body["threshold"] else "NORMAL")
    assert "no constituye un diagnóstico médico" in body["disclaimer"]


# ----------------------------------------------------------------------- errores HTTP

def test_rejects_non_image_content_type(client):
    r = client.post("/predict", files={"file": ("notas.txt", b"hola", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupted_image(client):
    r = client.post("/predict", files={"file": ("rx.jpeg", b"esto no es un jpeg" * 10, "image/jpeg")})
    assert r.status_code == 400
    assert "imagen" in r.json()["detail"]


def test_rejects_empty_file(client):
    r = client.post("/predict", files={"file": ("rx.png", b"", "image/png")})
    assert r.status_code == 400


def test_rejects_too_large_file(client):
    r = client.post("/predict", files={"file": ("rx.png", b"0" * (10 * 1024 * 1024 + 10), "image/png")})
    assert r.status_code == 413


def test_missing_model_returns_503(monkeypatch, tmp_path):
    with _make_client(monkeypatch, tmp_path / "no_existe.keras") as c:
        h = c.get("/health").json()
        assert h["model_loaded"] is False
        r = c.post("/predict", files={"file": ("rx.png", _image_bytes(), "image/png")})
        assert r.status_code == 503


def test_threshold_env_override(monkeypatch, tiny_model_path):
    # Con umbral 0.0 todo es PNEUMONIA; con umbral 1.01, todo es NORMAL.
    for thr, expected in [("0.0", "PNEUMONIA"), ("1.01", "NORMAL")]:
        with _make_client(monkeypatch, tiny_model_path, threshold=thr) as c:
            r = c.post("/predict", files={"file": ("rx.png", _image_bytes(), "image/png")})
            assert r.json()["prediction"] == expected


# ------------------------------------------------------------ modelo real (opcional)

REAL_MODEL = PROJECT_ROOT / "models" / "aixray_model.keras"
SPLIT_CSV = PROJECT_ROOT / "data" / "manifest_split.csv"


@pytest.mark.skipif(not (REAL_MODEL.exists() and SPLIT_CSV.exists()), reason="modelo real o manifiesto no disponibles")
def test_real_model_matches_offline_prediction(monkeypatch):
    """La API y el pipeline de evaluación offline deben dar la MISMA probabilidad."""
    import pandas as pd
    import tensorflow as tf

    from src.preprocessing import dataset_from_manifest

    split = pd.read_csv(SPLIT_CSV)
    rows = split[split["split"] == "test"].head(4)
    rows = rows[rows["path"].map(lambda p: Path(p).exists())]
    if rows.empty:
        pytest.skip("las imágenes del manifiesto no están en esta máquina")
    model = tf.keras.models.load_model(REAL_MODEL, compile=False)
    offline = model.predict(dataset_from_manifest(rows, "test", batch_size=4), verbose=0).ravel()

    with _make_client(monkeypatch, REAL_MODEL) as c:
        for path, p_off in zip(rows["path"], offline):
            mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
            r = c.post("/predict", files={"file": (Path(path).name, open(path, "rb").read(), mime)})
            assert r.status_code == 200
            assert abs(r.json()["probability"] - float(p_off)) < 1e-3
