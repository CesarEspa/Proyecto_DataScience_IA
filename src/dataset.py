"""
Construcción del manifiesto del dataset (rutas + etiquetas + metadatos) y splits.

El dataset original ("Pediatric Chest X-ray Pneumonia", Kermany et al. 2018, vía Kaggle)
viene repartido en dos carpetas, `train/` y `test/`, cada una con subcarpetas `NORMAL/` y
`PNEUMONIA/`. Ese split original NO se usa tal cual en este proyecto por dos razones,
explicadas también en el README:

1. No trae un set de validación utilizable (la versión típica de este dataset trae solo
   16 imágenes en `val/`, insuficiente para monitorear el entrenamiento).
2. El nombre de archivo de las imágenes de PNEUMONIA sigue el patrón
   `personNNNN_{bacteria|virus}_MMMM.jpeg`, donde `NNNN` identifica al paciente. Al
   inspeccionar el dataset se confirmó que el MISMO paciente puede aparecer tanto en
   `train/` como en `test/` (ej. `person100` tiene imágenes en ambas carpetas) -- eso es
   fuga de datos (*data leakage*) a nivel paciente si se entrena y evalúa con esas
   carpetas tal cual.

Por eso este módulo junta TODAS las imágenes (train + test originales) en un solo
manifiesto y hace su propio split 70/15/15, agrupando por paciente cuando el paciente es
identificable, para minimizar el riesgo de fuga.

Mejora respecto a la versión de clase: las imágenes NORMAL (`IM-XXXX-NNNN.jpeg` y
`NORMAL2-IM-XXXX-NNNN.jpeg`) también se agrupan por su prefijo `IM-XXXX`, que en el
dataset original identifica el estudio/paciente (varias imágenes comparten prefijo y
solo cambia el sufijo `-NNNN`). Antes cada imagen NORMAL era su propio "paciente".
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CLASSES, DATA_DIR, SEED, SPLIT_RATIOS

IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png"}
PATIENT_ID_RE = re.compile(r"^person(\d+)_")
NORMAL_ID_RE = re.compile(r"^(NORMAL2-)?IM-(\d+)-\d+")


def _extract_patient_id(filename: str, label_name: str, group_normal: bool = True) -> str:
    """Extrae un id de paciente del nombre de archivo cuando es posible.

    - PNEUMONIA: `personNNNN_bacteria_MMMM.jpeg` / `personNNNN_virus_MMMM.jpeg` -> "pneumonia_NNNN"
    - NORMAL: `IM-XXXX-NNNN.jpeg` -> "normal_IM-XXXX"; `NORMAL2-IM-XXXX-NNNN.jpeg` ->
      "normal_NORMAL2-IM-XXXX" (el prefijo NORMAL2 es otra serie, se mantiene separado).
      Con `group_normal=False` se reproduce el comportamiento original (1 imagen = 1 id).
    """
    if label_name == "PNEUMONIA":
        m = PATIENT_ID_RE.match(filename)
        if m:
            return f"pneumonia_{m.group(1)}"
    elif group_normal:
        m = NORMAL_ID_RE.match(filename)
        if m:
            return f"normal_{m.group(1) or ''}IM-{m.group(2)}"
    return f"{label_name.lower()}_{filename}"


def build_manifest(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Recorre `data_dir/{train,test}/{NORMAL,PNEUMONIA}` y arma un DataFrame con
    una fila por imagen: ruta absoluta, etiqueta, id de paciente y carpeta de origen.

    Nota: esta función usa pandas/`pathlib` (recorrido local). La versión pensada para
    "procesamiento masivo" (el paso que conecta esta clase con Databricks) es
    `build_manifest_spark`, más abajo -- aquí, con miles de imágenes, la diferencia de
    rendimiento es mínima, pero en un dataset real de cientos de miles de imágenes
    (como el NIH Chest X-ray14 completo) este recorrido se volvería el cuello de botella.
    """
    rows = []
    for origin in ["train", "test"]:
        for label_idx, label_name in enumerate(CLASSES):
            folder = data_dir / origin / label_name
            if not folder.exists():
                continue
            for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS):
                rows.append(
                    {
                        "path": str(path),
                        "filename": path.name,
                        "label": label_idx,
                        "label_name": label_name,
                        "patient_id": _extract_patient_id(path.name, label_name),
                        "origin_split": origin,  # split ORIGINAL del dataset, informativo
                    }
                )
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise FileNotFoundError(
            f"No se encontraron imágenes en {data_dir}. Revisa AIXRAY_DATA_DIR "
            "o data/README.md para confirmar dónde debe estar el dataset."
        )
    return manifest


def build_manifest_spark(data_dir: Path = DATA_DIR):
    """Equivalente a `build_manifest` usando PySpark, para el bloque de 'procesamiento
    masivo de datos' de la Clase 4. Con 5,856 imágenes esto es simbólico (no necesita un
    cluster), pero deja el código escrito como se escribiría con cientos de miles de
    imágenes: el mismo `spark.read.format("binaryFile")` que se usa aquí para leer y
    listar imágenes es el que se usaría, sin cambios, sobre una carpeta de un Data Lake.

    Devuelve un DataFrame de Spark con el mismo esquema (columnas) que `build_manifest`.
    Requiere PySpark instalado; si no está disponible, usar `build_manifest` (pandas).
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    spark = (
        SparkSession.builder.appName("AIXRay_Manifest")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    dfs = []
    for origin in ["train", "test"]:
        for label_name in CLASSES:
            folder = str(Path(data_dir) / origin / label_name)
            df = (
                spark.read.format("binaryFile")
                .option("pathGlobFilter", "*.{jpeg,jpg,png}")
                .load(folder)
                .select(
                    F.col("path"),
                    F.element_at(F.split(F.col("path"), "/"), -1).alias("filename"),
                    F.lit(CLASSES.index(label_name)).alias("label"),
                    F.lit(label_name).alias("label_name"),
                    F.lit(origin).alias("origin_split"),
                    F.col("length").alias("file_size_bytes"),
                )
            )
            dfs.append(df)

    manifest_sdf = dfs[0]
    for df in dfs[1:]:
        manifest_sdf = manifest_sdf.unionByName(df)

    # patient_id con expresiones NATIVAS de Spark (regexp_extract), no con una UDF de
    # Python: la UDF original importaba `src.dataset` dentro de los workers de Spark, que
    # no tienen `src` en su PYTHONPATH (ModuleNotFoundError). Además, las expresiones
    # nativas corren en la JVM, sin serializar filas a Python: es lo correcto a escala.
    fn, lbl = F.col("filename"), F.col("label_name")
    manifest_sdf = manifest_sdf.withColumn(
        "patient_id",
        F.when(
            (lbl == "PNEUMONIA") & fn.rlike(PATIENT_ID_RE.pattern),
            F.concat(F.lit("pneumonia_"), F.regexp_extract(fn, PATIENT_ID_RE.pattern, 1)),
        )
        .when(
            (lbl == "NORMAL") & fn.rlike(NORMAL_ID_RE.pattern),
            F.concat(
                F.lit("normal_"),
                F.regexp_extract(fn, NORMAL_ID_RE.pattern, 1),
                F.lit("IM-"),
                F.regexp_extract(fn, NORMAL_ID_RE.pattern, 2),
            ),
        )
        .otherwise(F.concat(F.lower(lbl), F.lit("_"), fn)),
    )
    return manifest_sdf


def make_balanced_subset(manifest: pd.DataFrame, per_class: int, seed: int = SEED) -> pd.DataFrame:
    """Toma `per_class` imágenes de cada clase (muestreo aleatorio con semilla fija) para
    el subconjunto de ~2,000 imágenes que se usa en la demo/laboratorio de clase.

    Se muestrea a nivel de PACIENTE (pacientes completos), no a nivel de imagen
    suelta, para no romper la agrupación por paciente antes de hacer el split.
    """
    parts = []
    rng = np.random.RandomState(seed)
    for label_idx, label_name in enumerate(CLASSES):
        subset = manifest[manifest["label"] == label_idx]
        patients = subset["patient_id"].unique()
        rng.shuffle(patients)

        chosen_rows, n_images = [], 0
        for pid in patients:
            if n_images >= per_class:  # se cuentan IMÁGENES (antes contaba pacientes)
                break
            rows = subset[subset["patient_id"] == pid]
            chosen_rows.append(rows)
            n_images += len(rows)
        # Descartar algunas imágenes sobrantes del último paciente no genera fuga: la
        # agrupación que importa es la del split (un paciente nunca en dos splits).
        class_subset = pd.concat(chosen_rows).sample(
            n=min(per_class, n_images), random_state=seed
        )
        parts.append(class_subset)

    result = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return result


def split_manifest(
    manifest: pd.DataFrame, ratios: dict = SPLIT_RATIOS, seed: int = SEED
) -> pd.DataFrame:
    """Split propio 70/15/15 (train/val/test) agrupado por `patient_id`, estratificado por
    clase. Un mismo paciente nunca queda repartido entre dos splits distintos.

    Devuelve el mismo manifiesto con una columna nueva `split` (train/val/test), en vez
    de tres DataFrames separados, para que sea fácil guardarlo como un solo CSV/tabla.
    """
    rng = np.random.RandomState(seed)
    manifest = manifest.copy()
    manifest["split"] = ""

    for label_idx in manifest["label"].unique():
        class_df = manifest[manifest["label"] == label_idx]
        patients = class_df["patient_id"].unique()
        rng.shuffle(patients)

        n = len(patients)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])

        train_patients = set(patients[:n_train])
        val_patients = set(patients[n_train : n_train + n_val])
        test_patients = set(patients[n_train + n_val :])

        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(train_patients)),
            "split",
        ] = "train"
        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(val_patients)), "split"
        ] = "val"
        manifest.loc[
            (manifest["label"] == label_idx) & (manifest["patient_id"].isin(test_patients)),
            "split",
        ] = "test"

    return manifest
