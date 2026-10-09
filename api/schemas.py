"""Esquemas Pydantic de la API -- documentan automáticamente el contrato en /docs."""

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    prediction: str = Field(..., description="Etiqueta predicha: 'NORMAL' o 'PNEUMONIA'.")
    probability: float = Field(
        ..., description="Probabilidad (0-1) de que la imagen sea PNEUMONIA, según el modelo."
    )
    threshold: float = Field(
        ..., description="Umbral de decisión aplicado (umbral clínico fijado con validación para sensibilidad >= 95%)."
    )
    model_version: str = Field(..., description="Versión del modelo que respondió.")
    disclaimer: str = Field(
        ..., description="Aviso obligatorio: este sistema es un proyecto académico, no un diagnóstico médico."
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str
    threshold: float


class ErrorResponse(BaseModel):
    detail: str
