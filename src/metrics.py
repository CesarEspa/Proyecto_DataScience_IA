"""
Evaluación del modelo más allá de accuracy: matriz de confusión, precision, recall
(sensibilidad), especificidad, F1, ROC-AUC y classification report -- clave en un
problema médico, donde un falso negativo (neumonía no detectada) no pesa lo mismo que
una falsa alarma.
"""

from pathlib import Path

import matplotlib.pyplot as plt  # sin forzar backend: así los notebooks muestran las figuras inline
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import CLASSES, DEFAULT_THRESHOLD


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Set completo de métricas en un dict plano, listo para `mlflow.log_metrics(...)`.
    Incluye especificidad (recall de NORMAL) y los conteos de la matriz de confusión."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if (tn + fp) else 0.0,
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def find_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, target_recall: float) -> float:
    """Umbral MÁS ALTO que todavía alcanza `target_recall` en el set dado (validación).
    El más alto = el que menos falsas alarmas genera cumpliendo el objetivo clínico.
    Se calcula SOLO con validación; luego se aplica tal cual al set de test."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    ok = np.where(tpr >= target_recall)[0]
    idx = ok[0]  # roc_curve ordena umbrales de mayor a menor: el primero que cumple
    return float(min(thresholds[idx], 1.0))


def classification_report_text(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> str:
    y_pred = (y_prob >= threshold).astype(int)
    return classification_report(y_true, y_pred, labels=[0, 1], target_names=CLASSES, zero_division=0, digits=3)


def save_confusion_matrix(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, threshold: float = DEFAULT_THRESHOLD, title: str = None
) -> Path:
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title or f"Matriz de confusión (umbral={threshold:.2f})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path, title: str = "Curva ROC") -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="AI-XRay", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar (AUC=0.50)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_training_curves(history, out_path: Path) -> Path:
    """Grafica loss/accuracy de entrenamiento vs. validación a partir del `history`
    (dict o el objeto que devuelve `model.fit(...)`)."""
    h = history.history if hasattr(history, "history") else history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, key, title in [(axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")]:
        ax.plot(h[key], label="train")
        ax.plot(h[f"val_{key}"], label="val")
        ax.set_title(title)
        ax.set_xlabel("Época")
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
