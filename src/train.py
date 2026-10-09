"""
Entrenamiento de una variante del modelo + registro completo del experimento en MLflow.

`train_and_log(...)` se llama una vez por variante desde `notebooks/03_training.ipynb`.
Cada llamada es un `mlflow.start_run()` independiente y comparable.

Mejoras respecto a la versión de clase:
1. Se evalúa en VALIDACIÓN (para elegir el mejor run) **y** en TEST (para reportar).
   Antes las métricas "finales" salían del mismo set de validación que usaban
   EarlyStopping / ReduceLROnPlateau y la selección de modelo -> eran optimistas.
2. Fine-tuning en dos fases real: `initial_model` permite continuar entrenando el
   modelo de la Fase A (cabeza ya entrenada) en vez de empezar con una cabeza aleatoria.
3. Umbral clínico: además del umbral 0.5 se calcula, SOLO con validación, el umbral que
   garantiza sensibilidad >= TARGET_RECALL y se reportan sus métricas en test.
4. Se loguean especificidad, conteos TP/FP/TN/FN, parámetros entrenables y tiempo.
"""

import json
import tempfile
import time
from pathlib import Path

import mlflow
import numpy as np
import tensorflow as tf
from mlflow.models import infer_signature
from sklearn.utils.class_weight import compute_class_weight

from src.architecture import (
    build_model,
    compile_model,
    count_trainable_params,
    set_seeds,
    unfreeze_for_finetuning,
)
from src.config import DEFAULT_THRESHOLD, IMG_CHANNELS, IMG_SIZE, SEED, TARGET_RECALL
from src.metrics import (
    classification_report_text,
    compute_metrics,
    find_threshold_for_recall,
    save_confusion_matrix,
    save_roc_curve,
    save_training_curves,
)


def predict_dataset(model: tf.keras.Model, ds: tf.data.Dataset):
    """Devuelve (y_true, y_prob) para un dataset de val/test (sin shuffle: el orden es
    determinista). Usa `model.predict` (grafo compilado): llamar `model(x)` en modo eager
    batch a batch dispara el consumo de RAM a ~6 GB en CPU con ResNet50."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    y_prob = model.predict(ds, verbose=0).ravel()
    return y_true, y_prob


def _prefixed(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}{k}": v for k, v in metrics.items()}


def evaluate_and_log(model, val_ds, test_ds, out_dir: Path, target_recall: float = TARGET_RECALL) -> dict:
    """Evalúa un modelo en val y test, con umbral 0.5 y con umbral clínico, y registra
    métricas + artefactos en el run de MLflow activo. Devuelve un resumen."""
    y_val, p_val = predict_dataset(model, val_ds)
    y_test, p_test = predict_dataset(model, test_ds)

    thr_clin = find_threshold_for_recall(y_val, p_val, target_recall)

    summary = {
        "val": compute_metrics(y_val, p_val, DEFAULT_THRESHOLD),
        "test": compute_metrics(y_test, p_test, DEFAULT_THRESHOLD),
        "val_clinical": compute_metrics(y_val, p_val, thr_clin),
        "test_clinical": compute_metrics(y_test, p_test, thr_clin),
        "threshold_clinical": thr_clin,
    }

    # Métricas "planas" (sin prefijo) = validación @0.5, para mantener compatibilidad con
    # la versión de clase (la selección del mejor run se hace sobre validación).
    mlflow.log_metrics(summary["val"])
    mlflow.log_metrics(_prefixed(summary["test"], "test_"))
    mlflow.log_metrics(_prefixed(summary["val_clinical"], "clin_val_"))
    mlflow.log_metrics(_prefixed(summary["test_clinical"], "clin_test_"))
    mlflow.log_metric("threshold_clinical", thr_clin)

    arts = [
        save_confusion_matrix(y_val, p_val, out_dir / "confusion_matrix.png",
                              title="Validación · umbral 0.50"),
        save_confusion_matrix(y_test, p_test, out_dir / "confusion_matrix_test.png",
                              title="Test · umbral 0.50"),
        save_confusion_matrix(y_test, p_test, out_dir / "confusion_matrix_test_clinical.png",
                              threshold=thr_clin, title=f"Test · umbral clínico {thr_clin:.3f}"),
        save_roc_curve(y_val, p_val, out_dir / "roc_curve.png", title="Curva ROC · validación"),
        save_roc_curve(y_test, p_test, out_dir / "roc_curve_test.png", title="Curva ROC · test"),
    ]
    report = (
        "=== VALIDACIÓN (umbral 0.50) ===\n" + classification_report_text(y_val, p_val)
        + "\n=== TEST (umbral 0.50) ===\n" + classification_report_text(y_test, p_test)
        + f"\n=== TEST (umbral clínico {thr_clin:.4f}, fijado con validación para recall>={target_recall}) ===\n"
        + classification_report_text(y_test, p_test, thr_clin)
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    arts.append(out_dir / "classification_report.txt")
    for a in arts:
        mlflow.log_artifact(str(a))
    np.savez(out_dir / "predictions.npz", y_val=y_val, p_val=p_val, y_test=y_test, p_test=p_test)
    mlflow.log_artifact(str(out_dir / "predictions.npz"))
    print(report)
    return summary


def train_and_log(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    test_ds: tf.data.Dataset,
    run_name: str,
    learning_rate: float,
    dropout: float,
    epochs: int = 10,
    fine_tune: bool = False,
    fine_tune_at: int = 143,
    initial_model: tf.keras.Model = None,
    parent_run_id: str = None,
    class_weight: dict = None,
    tags: dict = None,
    extra_params: dict = None,
    justification: str = None,
    weights: str = "imagenet",
    patience: int = 4,
):
    """Entrena una variante del modelo dentro de un run de MLflow y registra todo.

    - `fine_tune=False`: Fase A (base congelada, cabeza nueva).
    - `fine_tune=True` y `initial_model=None`: fine-tuning "en un solo paso" (como el
      run 3 de referencia de clase: cabeza aleatoria + base parcialmente descongelada).
    - `fine_tune=True` e `initial_model=<modelo de Fase A>`: fine-tuning en DOS FASES real.
    - `justification`: texto libre que queda como tag `justificacion` (la rúbrica pide
      variantes justificadas con criterio explícito).

    Devuelve `(run_id, model, summary)`.
    """
    set_seeds(SEED)

    if initial_model is not None:
        model = tf.keras.models.clone_model(initial_model)
        model.set_weights(initial_model.get_weights())
        if fine_tune:
            unfreeze_for_finetuning(model, fine_tune_at)
    else:
        model = build_model(
            dropout=dropout,
            base_trainable=fine_tune,
            fine_tune_at=fine_tune_at if fine_tune else None,
            weights=weights,
        )
    compile_model(model, learning_rate=learning_rate)

    with mlflow.start_run(run_name=run_name) as run:
        # -- 1. Parámetros ------------------------------------------------------------
        if not fine_tune:
            architecture = "ResNet50_frozen"
        elif initial_model is not None:
            architecture = "ResNet50_two_phase_fine_tuning"
        else:
            architecture = "ResNet50_fine_tuning"
        mlflow.log_params({
            "architecture": architecture,
            "learning_rate": learning_rate,
            "dropout": dropout,
            "epochs_max": epochs,
            "batch_size": train_ds.element_spec[0].shape[0] or "dynamic",
            "optimizer": "Adam",
            "fine_tune": fine_tune,
            "fine_tune_at_layer": fine_tune_at if fine_tune else None,
            "init_from_run": parent_run_id,
            "seed": SEED,
            "class_weight": class_weight,
            "early_stopping_patience": patience,
            "trainable_params": count_trainable_params(model),
        })
        if extra_params:
            mlflow.log_params(extra_params)
        all_tags = dict(tags or {})
        if justification:
            all_tags["justificacion"] = justification
        if all_tags:
            mlflow.set_tags(all_tags)

        # -- 2. Callbacks -------------------------------------------------------------
        tmp_dir = Path(tempfile.mkdtemp())
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(tmp_dir / f"{run_name}_best.keras"), monitor="val_loss", save_best_only=True
            ),
        ]

        # -- 3. Entrenamiento ---------------------------------------------------------
        t0 = time.time()
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=epochs,
            class_weight=class_weight, callbacks=callbacks, verbose=2,
        )
        train_seconds = time.time() - t0
        h = history.history
        for epoch in range(len(h["loss"])):
            mlflow.log_metrics(
                {
                    "epoch_loss": h["loss"][epoch],
                    "epoch_accuracy": h["accuracy"][epoch],
                    "epoch_val_loss": h["val_loss"][epoch],
                    "epoch_val_accuracy": h["val_accuracy"][epoch],
                    "epoch_val_auc": h["val_auc"][epoch],
                },
                step=epoch,
            )
        best_epoch = int(np.argmin(h["val_loss"]))
        mlflow.log_metrics({
            "epochs_run": len(h["loss"]),
            "best_epoch": best_epoch + 1,
            "best_val_loss": float(h["val_loss"][best_epoch]),
            "train_minutes": train_seconds / 60,
        })

        # -- 4. Evaluación completa: val (selección) + test (reporte) -----------------
        save_training_curves(history, tmp_dir / "training_curves.png")
        mlflow.log_artifact(str(tmp_dir / "training_curves.png"))
        mlflow.log_dict({k: [float(v) for v in vals] for k, vals in h.items()}, "history.json")
        summary = evaluate_and_log(model, val_ds, test_ds, tmp_dir)

        # -- 5. El modelo, con signature ----------------------------------------------
        sample_input = np.zeros((1, *IMG_SIZE, IMG_CHANNELS), dtype="float32")
        signature = infer_signature(sample_input, model.predict(sample_input, verbose=0))
        # Workaround verificado (MLflow 3.x + Keras 3): `input_example` en log_model
        # falla con FileNotFoundError; se documenta la entrada como artefacto JSON aparte.
        model_info = mlflow.keras.log_model(model, name="model", signature=signature)
        mlflow.log_dict(
            {
                "input_shape": list(sample_input.shape),
                "dtype": "float32",
                "preprocess": "tf.image.resize(224x224, bilinear) + resnet50.preprocess_input",
                "threshold_default": DEFAULT_THRESHOLD,
                "threshold_clinical": summary["threshold_clinical"],
            },
            "input_example_info.json",
        )
        mlflow.set_tag("logged_model_uri", model_info.model_uri)
        run_id = run.info.run_id

    print(
        f"Run '{run_name}' -- run_id={run_id} -- {len(h['loss'])} épocas, {train_seconds/60:.1f} min\n"
        f"  VAL  @0.5: {json.dumps({k: round(v, 4) for k, v in summary['val'].items()})}\n"
        f"  TEST @0.5: {json.dumps({k: round(v, 4) for k, v in summary['test'].items()})}"
    )
    return run_id, model, summary


def compute_balanced_class_weight(labels) -> dict:
    """Calcula `class_weight` ({0: peso, 1: peso}) para compensar desbalance de clases.
    En el subconjunto balanceado da ~1.0/1.0; su utilidad real aparece con el dataset
    completo (desbalanceado ~2.7:1 a favor de PNEUMONIA)."""
    labels = np.asarray(labels)
    classes = np.unique(labels)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return {int(c): float(w) for c, w in zip(classes, weights)}
