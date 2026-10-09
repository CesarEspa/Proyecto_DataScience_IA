"""
Reubica el tracking de MLflow (mlflow.db + mlruns/) cuando la carpeta AI-XRay se mueve
de lugar o de máquina.

Por qué hace falta: el backend SQLite de MLflow guarda RUTAS ABSOLUTAS a los artefactos
(`file:///ruta/original/AI-XRay/mlruns/...`). Si copias el proyecto a otra carpeta u otro
computador, `mlflow ui` sigue mostrando runs y métricas, pero no encuentra los artefactos
ni puede cargar los modelos del Model Registry. Este script reescribe esas rutas para que
apunten a la ubicación ACTUAL del proyecto.

Uso (desde la carpeta AI-XRay/):
    python scripts/relocate_mlflow.py

Es idempotente: si ya apunta a la carpeta actual, no cambia nada. Hace una copia de
seguridad `mlflow.db.bak` antes de modificar.
"""

import shutil
import sqlite3
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB = PROJECT_ROOT / "mlflow.db"


def _uri_to_root(uri: str) -> str:
    """'file:///x/AI-XRay/mlruns/1' -> '/x/AI-XRay' (string tal como está en la BD)."""
    marker = "/mlruns"
    return uri[: uri.index(marker)] if marker in uri else None


def main():
    if not DB.exists():
        sys.exit(f"No existe {DB}. ¿Estás en la carpeta AI-XRay?")
    con = sqlite3.connect(DB)
    row = con.execute("SELECT artifact_location FROM experiments WHERE name = 'AI-XRay'").fetchone()
    if not row:
        sys.exit("No se encontró el experiment 'AI-XRay' en mlflow.db")

    old_uri_root = _uri_to_root(row[0])  # p. ej. file:///ruta/anterior/AI-XRay
    new_uri_root = PROJECT_ROOT.as_uri()  # p. ej. file:///C:/Users/.../AI-XRay
    if old_uri_root == new_uri_root:
        print("MLflow ya apunta a esta carpeta. Nada que hacer.")
        return
    old_path_root = unquote(urlparse(old_uri_root).path)
    new_path_root = PROJECT_ROOT.as_posix()

    shutil.copy2(DB, DB.with_suffix(".db.bak"))
    total = 0
    tables = [t for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    for table in tables:
        cols = [c[1] for c in con.execute(f'PRAGMA table_info("{table}")') if "CHAR" in c[2].upper() or "TEXT" in c[2].upper()]
        for col in cols:
            for old, new in [(old_uri_root, new_uri_root), (old_path_root, new_path_root)]:
                cur = con.execute(
                    f'UPDATE "{table}" SET "{col}" = REPLACE("{col}", ?, ?) WHERE "{col}" LIKE ?',
                    (old, new, f"%{old}%"),
                )
                total += cur.rowcount
    con.commit()
    con.close()
    print(f"Rutas actualizadas: {total} valores")
    print(f"  antes:   {old_uri_root}")
    print(f"  después: {new_uri_root}")
    print("Copia de seguridad: mlflow.db.bak")


if __name__ == "__main__":
    main()
