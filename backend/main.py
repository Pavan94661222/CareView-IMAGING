"""FastAPI application exposing the pneumonia detection model.

Run it with::

    uvicorn backend.main:app --reload --port 8000

Endpoints
---------
``GET  /api/health``   service + model status
``GET  /api/model``    metadata of the loaded model
``GET  /api/metrics``  current model performance report
``POST /api/predict``  multipart upload -> NORMAL / PNEUMONIA + confidence
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import tensorflow as tf
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from pneumonia import config
from pneumonia.inference import ModelNotAvailableError, PneumoniaPredictor

from . import __version__ as API_VERSION
from . import clinical_store
from .auth_routes import router as auth_router
from .clinical_routes import router as clinical_router
from .dependencies import get_predictor
from .schemas import (
    ErrorResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfo,
    PredictionResponse,
)
from .service import run_prediction
from .validation import UploadValidationError

logger = logging.getLogger("pneumonia.api")


def _model_info(predictor: PneumoniaPredictor) -> ModelInfo:
    info = predictor.info()
    return ModelInfo(
        name=info["model_name"],
        backbone=info["backbone"],
        img_size=info["img_size"],
        class_names=info["class_names"],
        positive_class=info["positive_class"],
        threshold=info["threshold"],
        created_at=info.get("created_at"),
    )


# --------------------------------------------------------------------------
# Lifespan: warm the model up so the first request is not slow.
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    clinical_store.initialize_database()
    predictor = get_predictor()
    if predictor.is_ready:
        logger.info(
            "Model ready: %s (%s) threshold=%.2f",
            predictor.model_path.name,
            predictor.backbone,
            predictor.threshold,
        )
    else:
        logger.warning(
            "Model NOT loaded - %s. Train it with 'python -m pneumonia.train' or set "
            "PNEUMONIA_MODEL_PATH.",
            predictor.health().get("error"),
        )
    yield
    logger.info("Shutting down pneumonia detection API.")


app = FastAPI(
    title="Pneumonia Detection API",
    description=(
        "Classifies chest X-ray images as NORMAL or PNEUMONIA using a "
        "transfer-learning CNN. Preprocessing is identical to the training pipeline."
    ),
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-CSRF-Token"],
)
app.include_router(auth_router)
app.include_router(clinical_router)


@app.middleware("http")
async def add_private_api_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


# --------------------------------------------------------------------------
# Exception handlers -> consistent JSON error contract
# --------------------------------------------------------------------------
@app.exception_handler(UploadValidationError)
async def handle_upload_error(request: Request, exc: UploadValidationError):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.message, code=exc.code).model_dump(),
    )


@app.exception_handler(ModelNotAvailableError)
async def handle_model_unavailable(request: Request, exc: ModelNotAvailableError):
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=ErrorResponse(error=str(exc), code="model_unavailable").model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="The request could not be validated.",
            code="validation_error",
            detail=json.dumps(exc.errors(), default=str),
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def handle_unexpected(request: Request, exc: Exception):  # pragma: no cover
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="An unexpected server error occurred while processing the image.",
            code="internal_error",
        ).model_dump(),
    )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "Pneumonia Detection API",
        "version": API_VERSION,
        "docs": "/docs",
        "endpoints": [
            "/api/health",
            "/api/model",
            "/api/metrics",
            "/api/predict",
            "/api/auth/*",
            "/api/dashboard",
            "/api/patients",
            "/api/analyses",
        ],
    }


@app.get("/api/health", response_model=HealthResponse, tags=["system"])
async def health(predictor: PneumoniaPredictor = Depends(get_predictor)) -> HealthResponse:
    state = predictor.health()
    return HealthResponse(
        status="ok" if state["ready"] else "degraded",
        model_ready=bool(state["ready"]),
        detail=state.get("error"),
        model=_model_info(predictor) if state["ready"] else None,
        tensorflow_version=tf.__version__,
        api_version=API_VERSION,
    )


@app.get("/api/model", response_model=ModelInfo, tags=["system"])
async def model_info(predictor: PneumoniaPredictor = Depends(get_predictor)) -> ModelInfo:
    if not predictor.is_ready:
        raise ModelNotAvailableError(predictor.health().get("error") or "Model not loaded.")
    return _model_info(predictor)


@app.get("/api/metrics", response_model=MetricsResponse, tags=["system"])
async def metrics() -> MetricsResponse:
    """Return the current model performance report, if available."""
    report_path = Path(config.REPORTS_DIR) / "test_metrics.json"
    if not report_path.is_file():
        return MetricsResponse(
            available=False,
            detail=f"No model performance report found at {report_path}.",
        )
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            return MetricsResponse(available=True, report=json.load(handle))
    except (json.JSONDecodeError, OSError) as exc:
        return MetricsResponse(available=False, detail=f"Could not read the report: {exc}")


@app.post("/api/predict", response_model=PredictionResponse, tags=["prediction"])
async def predict(
    file: UploadFile = File(..., description="Chest X-ray image (JPEG/PNG/BMP/TIFF)"),
    predictor: PneumoniaPredictor = Depends(get_predictor),
) -> PredictionResponse:
    """Classify a chest X-ray as NORMAL or PNEUMONIA and return the confidence."""
    data = await file.read()
    payload = run_prediction(
        predictor,
        data,
        filename=file.filename,
        content_type=file.content_type,
    )
    return PredictionResponse(**payload)
