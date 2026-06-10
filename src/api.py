"""
API REST para clasificación de riesgo agrícola con CatBoost.

Levantar localmente:
    uvicorn src.api:app --reload --host 0.0.0.0 --port 8000

Ejemplo de predicción:
    curl -X POST http://localhost:8000/predict \\
         -H "Content-Type: application/json" \\
         -d '{
               "Nomestado": "OAXACA",
               "Nomcultivo": "Maíz grano",
               "Nomcicloproductivo": "P-V",
               "Nommodalidad": "Temporal",
               "Sembrada": 2.5,
               "Precio": 4200.0
             }'
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Ruta al modelo serializado (relativa al directorio de trabajo del proceso)
_MODEL_PATH = Path(os.getenv("CATBOOST_MODEL_PATH", "models/catboost_model.pkl"))

# Estado global de la aplicación (cargado una sola vez al arrancar)
_state: dict = {"trainer": None, "model_loaded": False}


# ---------------------------------------------------------------------------
# Lifespan: carga del modelo al arrancar la aplicación
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carga el modelo al iniciar y libera recursos al cerrar."""
    try:
        # Importación aquí para no forzar la dependencia en todos los contextos
        from src.catboost_model import CatBoostTrainer

        trainer = CatBoostTrainer.cargar(_MODEL_PATH)
        _state["trainer"] = trainer
        _state["model_loaded"] = True
        print(f"[API] Modelo CatBoost cargado desde {_MODEL_PATH}")
    except FileNotFoundError:
        print(f"[API] ADVERTENCIA: No se encontró el modelo en {_MODEL_PATH}. "
              f"El endpoint /predict no estará disponible hasta que se entrene "
              f"y serialice el modelo.")
        _state["model_loaded"] = False
    yield
    # Limpieza al cerrar (si fuera necesaria)
    _state["trainer"] = None
    _state["model_loaded"] = False


# ---------------------------------------------------------------------------
# Aplicación FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="API de Riesgo Agrícola — CatBoost",
    description=(
        "Clasifica el nivel de riesgo de siniestro de una siembra mexicana "
        "usando solo información disponible **antes** de la cosecha. "
        "Modelo: CatBoost con ordered boosting y manejo nativo de categóricas. "
        "Dataset: SIAP 2010–2024."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Esquemas Pydantic
# ---------------------------------------------------------------------------

NIVELES_RIESGO = {
    0: "Sin siniestro",
    1: "Riesgo bajo",
    2: "Riesgo medio",
    3: "Riesgo alto",
    4: "Pérdida crítica",
}


class RegistroAgricola(BaseModel):
    """Variables disponibles ANTES de la cosecha (sin data leakage post-cosecha).

    Excluidas deliberadamente: Cosechada, Volumenproduccion, Rendimiento,
    Siniestrada, Valorproduccion — esas columnas se conocen DESPUÉS de la
    cosecha y usarlas en predicción constituiría data leakage.
    """

    Nomestado: str = Field(
        ...,
        description="Nombre del estado (p. ej. 'OAXACA', 'JALISCO').",
        examples=["OAXACA"],
    )
    Nomcultivo: str = Field(
        ...,
        description="Nombre del cultivo (p. ej. 'Maíz grano', 'Frijol').",
        examples=["Maíz grano"],
    )
    Nomcicloproductivo: str = Field(
        ...,
        description="Ciclo productivo (p. ej. 'P-V', 'O-I').",
        examples=["P-V"],
    )
    Nommodalidad: str = Field(
        ...,
        description="Modalidad de cultivo: 'Riego' o 'Temporal'.",
        examples=["Temporal"],
    )
    Sembrada: float = Field(
        ...,
        ge=0.0,
        description="Superficie sembrada en hectáreas.",
        examples=[2.5],
    )
    Precio: float = Field(
        ...,
        ge=0.0,
        description="Precio estimado por tonelada al momento de la siembra (pesos MXN).",
        examples=[4200.0],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "Nomestado": "OAXACA",
                "Nomcultivo": "Maíz grano",
                "Nomcicloproductivo": "P-V",
                "Nommodalidad": "Temporal",
                "Sembrada": 2.5,
                "Precio": 4200.0,
            }
        }
    }


class RespuestaPrediccion(BaseModel):
    """Respuesta del endpoint /predict."""

    nivel_riesgo: str = Field(
        ...,
        description="Nivel de riesgo predicho: 'Sin siniestro', 'Riesgo bajo', "
                    "'Riesgo medio', 'Riesgo alto' o 'Pérdida crítica'.",
    )
    probabilidades: Dict[str, float] = Field(
        ...,
        description="Probabilidad estimada para cada clase (suma = 1.0).",
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", summary="Estado del servicio", tags=["Salud"])
def root() -> dict:
    """Verifica que el servicio está en línea."""
    return {"status": "ok", "model": "CatBoost Agricultural Risk"}


@app.get("/health", summary="Estado de salud del servicio", tags=["Salud"])
def health() -> dict:
    """Devuelve si el modelo está cargado en memoria."""
    return {
        "status": "ok",
        "model_loaded": _state["model_loaded"],
    }


@app.post(
    "/predict",
    response_model=RespuestaPrediccion,
    summary="Predicción de riesgo agrícola",
    tags=["Predicción"],
)
def predict(registro: RegistroAgricola) -> RespuestaPrediccion:
    """Clasifica el nivel de riesgo de una siembra mexicana.

    Recibe las características de la siembra disponibles **antes** de la
    cosecha y devuelve:
    - La clase de riesgo predicha (etiqueta textual).
    - Las probabilidades estimadas para cada una de las 5 clases.

    El modelo CatBoost maneja las variables categóricas (Nomestado,
    Nomcultivo, Nomcicloproductivo, Nommodalidad) de forma nativa,
    sin encoding previo por parte del cliente.
    """
    if not _state["model_loaded"] or _state["trainer"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "El modelo no está disponible. Entrena y serializa el modelo "
                "ejecutando el notebook modelo_supervisado.ipynb y luego "
                "reinicia el servicio."
            ),
        )

    trainer = _state["trainer"]

    # Construir el DataFrame con proporcion_siniestro = 0 (valor nulo en siembra)
    # Esto es correcto: antes de la cosecha no se conoce la proporción de siniestro.
    X = pd.DataFrame(
        [
            {
                "Nomestado": registro.Nomestado,
                "Nomcicloproductivo": registro.Nomcicloproductivo,
                "Nommodalidad": registro.Nommodalidad,
                "Nomcultivo": registro.Nomcultivo,
                "Sembrada": float(registro.Sembrada),
                "Precio": float(registro.Precio),
                "proporcion_siniestro": 0.0,
            }
        ]
    )

    try:
        clase_pred = int(trainer.predecir(X)[0])
        probas = trainer.predecir_proba(X)[0]
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error interno al ejecutar la predicción: {exc}",
        )

    probabilidades = {
        NIVELES_RIESGO[i]: float(round(probas[i], 6))
        for i in range(5)
    }

    return RespuestaPrediccion(
        nivel_riesgo=NIVELES_RIESGO[clase_pred],
        probabilidades=probabilidades,
    )
