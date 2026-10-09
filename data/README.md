# Dataset -- dónde debe estar y cómo se usa

## Dataset

**Pediatric Chest X-ray Pneumonia** (Kaggle, usuario `andrewmvd`):
https://www.kaggle.com/datasets/andrewmvd/pediatric-pneumonia-chest-xray

5,856 radiografías de tórax pediátricas, etiquetadas `NORMAL` o `PNEUMONIA`.

**Cita obligatoria** (incluirla también en el Executive Summary):

> Kermany D, Goldbaum M, Cai W, et al. *Identifying Medical Diagnoses and Treatable
> Diseases by Image-Based Deep Learning*. Cell. 2018;172(5):1122-1131.
> doi:10.1016/j.cell.2018.02.010
>
> Kermany, Daniel; Zhang, Kang; Goldbaum, Michael (2018), "Labeled Optical Coherence
> Tomography (OCT) and Chest X-Ray Images for Classification", Mendeley Data, v2.
> http://dx.doi.org/10.17632/rscbjbr9sj.2

Uso estrictamente académico. **Las imágenes no se suben al repositorio** (`.gitignore`
ya excluye `archive*/` y los `.csv` generados en `data/`); son ~1.2 GB.

## Dónde colocar el dataset

El proyecto espera el dataset descomprimido un nivel arriba de esta carpeta:

```text
clase_04_laboratorio_practico/
├── archive (2)/Pediatric Chest X-ray Pneumonia/{train,test}/{NORMAL,PNEUMONIA}
└── AI-XRay/                    <- este proyecto
```

`src/config.py` ya apunta ahí por defecto (`DEFAULT_DATA_DIR`). No hace falta mover
nada para correr los notebooks tal cual.

Si prefieres tener las imágenes en otro lugar (por ejemplo, si subes este proyecto a tu
propio repositorio y no quieres depender de una ruta relativa "hacia arriba"), colócalas
donde quieras y define la variable de entorno antes de correr los notebooks:

```bash
export AIXRAY_DATA_DIR="/ruta/a/tu/copia/del/dataset"
```

(en Windows/PowerShell: `$env:AIXRAY_DATA_DIR = "C:\ruta\a\tu\copia\del\dataset"`)

Descarga con `kagglehub` (requiere `~/.kaggle/kaggle.json`):

```python
import kagglehub, shutil
from pathlib import Path
src = Path(kagglehub.dataset_download("andrewmvd/pediatric-pneumonia-chest-xray"))
shutil.copytree(src, Path("..") / "archive (2)", dirs_exist_ok=True)  # desde AI-XRay/
```

Además de este `README.md`, esta carpeta contiene los archivos `manifest.csv` y
`manifest_split.csv`, que genera `01_exploracion.ipynb` (se incluyen los de esta instancia
para trazabilidad; si los regeneras en otra máquina, las rutas absolutas cambian).

## Composición real (confirmada)

| Carpeta original | NORMAL | PNEUMONIA | Total |
|---|---|---|---|
| `train/` | 1,349 | 3,883 | 5,232 |
| `test/` | 234 | 390 | 624 |
| **Total** | **1,583** | **4,273** | **5,856** |

## Por qué este proyecto NO usa el split `train/`/`test/` original tal cual

1. **No trae un set de validación usable.** La versión típica de este dataset separa
   solo 16 imágenes para validación -- insuficiente para monitorear el entrenamiento
   con `EarlyStopping`/`ReduceLROnPlateau`.
2. **Fuga de datos a nivel paciente (confirmada).** Las imágenes de `PNEUMONIA` siguen
   el patrón `personNNNN_{bacteria|virus}_MMMM.jpeg`, donde `NNNN` identifica al
   paciente. Al inspeccionar el dataset se confirmó que el mismo paciente puede tener
   imágenes tanto en `train/` como en `test/` (ejemplo real encontrado: `person100`
   tiene imágenes en ambas carpetas). Entrenar y evaluar con las carpetas originales
   sobreestimaría el desempeño real del modelo.

Por eso `src/dataset.py` junta TODAS las imágenes (train + test originales) y
genera su propio split 70/15/15, agrupado por paciente cuando el paciente es
identificable (ver `split_manifest()` y el chequeo de fuga en
`notebooks/01_exploracion.ipynb`).

**Mejora aplicada en esta instancia (NORMAL también se agrupa):** las imágenes `NORMAL`
se llaman `IM-XXXX-NNNN.jpeg` o `NORMAL2-IM-XXXX-NNNN.jpeg`. Al inspeccionarlas se comprobó
que el prefijo `IM-XXXX` agrupa varias radiografías del mismo estudio (75 estudios con más
de una imagen, 214 imágenes; p. ej. `IM-0553-0001`, `IM-0553-0001-0001`, `-0002`, `-0003`).
Ahora `split_manifest()` agrupa también por ese prefijo. Con la agrupación original, 29 de
esos estudios habrían quedado repartidos entre train/val/test en el subconjunto de la demo
(ver `notebooks/01_exploracion.ipynb`). Sigue siendo una aproximación: el prefijo identifica
el estudio, no garantiza que dos prefijos distintos sean pacientes distintos.

## Subconjunto usado en la demo de clase

Por rapidez, el laboratorio entrena con un subconjunto de **2,000 imágenes balanceadas**
(1,000 NORMAL + 1,000 PNEUMONIA), seleccionadas por paciente/estudio completo,
muestreadas con semilla fija (`SEED=42` en `src/config.py`) para que los resultados
sean reproducibles entre estudiantes. El dataset completo (5,856 imágenes, desbalanceado
~2.7:1 a favor de PNEUMONIA) queda disponible como extensión natural del proyecto --
en ese caso, usar `compute_balanced_class_weight()` (`src/train.py`) para
compensar el desbalance en vez de volver a submuestrear.
