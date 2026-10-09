"""
AI-XRay -- API de inferencia (FastAPI).

Backend que carga el modelo ya entrenado y expone un endpoint para clasificar una
radiografía. No hay frontend: se prueba en `/docs` (Swagger), con `curl` o con
`pytest tests/ -v`.

Levantar en local (desde la carpeta AI-XRay/):
    uvicorn api.main:app --reload --port 8000
Luego abrir: http://localhost:8000/docs

Manejo de errores (rúbrica: "códigos HTTP correctos"):
    400  archivo vacío o que no es una imagen decodificable
    413  archivo demasiado grande (> 10 MB)
    415  tipo de archivo no soportado (no es JPEG/PNG)
    503  el modelo no está disponible (no se ha entrenado/exportado)
    500  error interno inesperado durante la inferencia
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from api.model_service import InvalidImageError, ModelNotAvailableError, model_service
from api.schemas import ErrorResponse, HealthResponse, PredictionResponse

DISCLAIMER = (
    "Este sistema es un proyecto académico de inteligencia artificial y no constituye "
    "un diagnóstico médico ni sustituye la evaluación de un profesional de la salud."
)
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB: una radiografía JPEG típica pesa < 1 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar el modelo UNA sola vez al iniciar el proceso, no en cada petición.
    try:
        model_service.load()
    except ModelNotAvailableError as e:
        # No tumbamos el proceso: /health reporta model_loaded=false y /predict responde 503.
        print(f"[startup] Aviso: {e}")
    yield


app = FastAPI(
    title="AI-XRay -- API de inferencia",
    description=(
        "Clasificación de radiografías de tórax pediátricas (NORMAL / PNEUMONIA) con "
        "ResNet50 + Transfer Learning. Proyecto académico. " + DISCLAIMER
    ),
    version="1.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["monitoreo"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded(),
        model_version=model_service.version_label,
        threshold=model_service.threshold,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse}, 413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
        500: {"model": ErrorResponse}, 503: {"model": ErrorResponse},
    },
    tags=["inferencia"],
)
def predict(file: UploadFile = File(..., description="Imagen de radiografía (JPEG/PNG)")):
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Tipo de archivo no soportado: {file.content_type}. Usa JPEG o PNG.",
        )

    image_bytes = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="El archivo está vacío.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera el máximo de 10 MB.")

    try:
        result = model_service.predict_image(image_bytes)
    except ModelNotAvailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except InvalidImageError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # cualquier otro fallo es del servidor, no del cliente
        raise HTTPException(status_code=500, detail=f"Error interno durante la inferencia: {type(e).__name__}")

    return {**result, "disclaimer": DISCLAIMER}
