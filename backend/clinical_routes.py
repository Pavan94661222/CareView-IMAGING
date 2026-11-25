"""Authenticated, owner-scoped patient and analysis APIs."""

from __future__ import annotations

import os
import re
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, Response

from pneumonia import config
from pneumonia.inference import PneumoniaPredictor

from . import clinical_store
from .auth_security import get_current_user, get_current_user_csrf, require_same_origin
from .dependencies import get_predictor
from .explanations import local_explanation
from .pdf_reports import build_analysis_pdf
from .service import run_prediction
from .validation import UploadValidationError

router = APIRouter(prefix="/api", tags=["clinical records"])
SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def _validate_text(value: Optional[str], field: str, max_length: int) -> str:
    normalized = (value or "").strip()
    if len(normalized) > max_length:
        raise HTTPException(status_code=422, detail=f"{field} cannot exceed {max_length} characters.")
    return normalized


def _public_analysis(record: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(record)
    result.pop("image_path", None)
    result["image_url"] = f"/api/analyses/{result['id']}/image"
    result["report_url"] = f"/api/analyses/{result['id']}/report.pdf"
    return result


def _stored_image_path(value: str) -> Path:
    root = (clinical_store.data_dir() / "images").resolve()
    path = Path(value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Image not found.") from exc
    return path


def _remove_image(value: str) -> None:
    path = _stored_image_path(value)
    if path.is_file():
        path.unlink()


def _safe_filename(value: Optional[str]) -> str:
    leaf = (value or "chest-xray").replace("\\", "/").rsplit("/", 1)[-1]
    sanitized = SAFE_FILENAME.sub("_", leaf).strip("._-")
    return (sanitized or "chest-xray")[:180]


@router.get("/dashboard")
def dashboard(user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    summary = clinical_store.dashboard_summary(user["id"])
    summary["recent_analyses"] = [_public_analysis(record) for record in summary["recent_analyses"]]
    return summary


@router.post(
    "/analyses",
    status_code=201,
    dependencies=[Depends(require_same_origin)],
)
async def create_analysis(
    file: UploadFile = File(...),
    patient_code: Optional[str] = Form(None),
    patient_name: Optional[str] = Form(None),
    age: Optional[int] = Form(None, ge=0, le=130),
    sex: Optional[str] = Form(None),
    symptoms: Optional[str] = Form(None),
    clinical_history: Optional[str] = Form(None),
    notes: Optional[str] = Form(None),
    user: Dict[str, Any] = Depends(get_current_user_csrf),
    predictor: PneumoniaPredictor = Depends(get_predictor),
) -> Dict[str, Any]:
    data = await file.read(config.MAX_UPLOAD_BYTES + 1)
    payload = run_prediction(predictor, data, filename=file.filename, content_type=file.content_type)
    patient_code = _validate_text(patient_code, "Patient ID", 80) or None
    patient_name = _validate_text(patient_name, "Patient name", 120) or None
    image_filename = _safe_filename(file.filename)
    payload["image"]["filename"] = image_filename
    symptoms = _validate_text(symptoms, "Symptoms", 2000)
    clinical_history = _validate_text(clinical_history, "Clinical history", 4000)
    notes = _validate_text(notes, "Notes", 4000)
    sex = _validate_text(sex, "Sex", 40) or None
    if age is not None and not 0 <= age <= 130:
        raise HTTPException(status_code=422, detail="Age must be between 0 and 130.")
    if sex and sex not in {"Female", "Male", "Other", "Prefer not to say"}:
        raise HTTPException(status_code=422, detail="Select a valid sex value.")
    if any((symptoms, clinical_history)):
        payload["explanation"] = local_explanation(payload, symptoms, clinical_history)
    else:
        payload["explanation"] = local_explanation(payload)
    payload["privacy_notice"] = (
        "Image processing and storage happen locally on this server. Patient images and details "
        "are not sent to Gemini or another external AI service."
    )

    image_format = str(payload["image"]["format"]).lower()
    extension = "jpg" if image_format in {"jpeg", "jpg"} else image_format
    analysis_id = uuid.uuid4().hex
    target_dir = clinical_store.data_dir() / "images" / user["id"]
    target_dir.mkdir(parents=True, exist_ok=True)
    image_path = target_dir / f"{analysis_id}.{extension}"
    with image_path.open("xb") as handle:
        handle.write(data)

    try:
        clinical_store.create_analysis(
            analysis_id=analysis_id,
            user_id=user["id"],
            image_path=image_path,
            image_filename=image_filename,
            image_width=payload["image"]["width"],
            image_height=payload["image"]["height"],
            result=payload,
            patient_code=patient_code,
            patient_name=patient_name,
            age=age,
            sex=sex,
            symptoms=symptoms,
            clinical_history=clinical_history,
            notes=notes,
        )
    except sqlite3.IntegrityError as exc:
        image_path.unlink(missing_ok=True)
        raise HTTPException(status_code=409, detail="That patient ID is already in use.") from exc
    except Exception:
        image_path.unlink(missing_ok=True)
        raise
    record = clinical_store.get_analysis(user["id"], analysis_id)
    if not record:
        _remove_image(str(image_path))
        raise HTTPException(status_code=500, detail="Analysis was saved but could not be loaded.")
    return _public_analysis(record)


@router.get("/analyses")
def analyses(
    q: str = Query("", max_length=120),
    prediction: str = Query("", pattern="^(|NORMAL|PNEUMONIA)$"),
    order: str = Query("newest", pattern="^(newest|oldest|patient|prediction)$"),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    records, total = clinical_store.list_analyses(
        user["id"], query=q.strip(), prediction=prediction, order=order, limit=limit, offset=offset
    )
    return {
        "items": [_public_analysis(record) for record in records],
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@router.get("/analyses/{analysis_id}")
def analysis_detail(
    analysis_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    record = clinical_store.get_analysis(user["id"], analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    return _public_analysis(record)


@router.get("/analyses/{analysis_id}/image")
def analysis_image(
    analysis_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> FileResponse:
    record = clinical_store.get_analysis(user["id"], analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    path = _stored_image_path(record["image_path"])
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".bmp": "image/bmp",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }.get(path.suffix.lower(), "application/octet-stream")
    return FileResponse(path, media_type=media_type, filename=record["image_filename"])


@router.get("/analyses/{analysis_id}/report.pdf")
def analysis_report(
    analysis_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    record = clinical_store.get_analysis(user["id"], analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    filename = f"analysis-{analysis_id[:8]}.pdf"
    return Response(
        content=build_analysis_pdf(record),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/analyses/{analysis_id}", status_code=204, dependencies=[Depends(require_same_origin)])
def remove_analysis(
    analysis_id: str,
    user: Dict[str, Any] = Depends(get_current_user_csrf),
) -> Response:
    image_path = clinical_store.delete_analysis(user["id"], analysis_id)
    if not image_path:
        raise HTTPException(status_code=404, detail="Analysis not found.")
    _remove_image(image_path)
    return Response(status_code=204)


@router.get("/patients")
def patients(
    q: str = Query("", max_length=120),
    user: Dict[str, Any] = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    return clinical_store.list_patients(user["id"], q.strip())


@router.get("/patients/{patient_id}")
def patient_detail(
    patient_id: str,
    user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    patient = clinical_store.get_patient(user["id"], patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found.")
    patient["analyses"] = [
        _public_analysis(record)
        for record in clinical_store.list_patient_analyses(user["id"], patient_id)
    ]
    return patient


@router.delete("/patients/{patient_id}", status_code=204, dependencies=[Depends(require_same_origin)])
def remove_patient(
    patient_id: str,
    user: Dict[str, Any] = Depends(get_current_user_csrf),
) -> Response:
    image_paths = clinical_store.delete_patient(user["id"], patient_id)
    if image_paths is None:
        raise HTTPException(status_code=404, detail="Patient not found.")
    for path in image_paths:
        _remove_image(path)
    return Response(status_code=204)
