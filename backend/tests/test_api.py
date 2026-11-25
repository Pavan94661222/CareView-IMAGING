"""Integration tests for the FastAPI backend.

A very small, randomly initialised EfficientNetB0 is trained-free and saved to a
temporary folder, so the tests exercise the *whole* HTTP -> validation ->
preprocessing -> model -> JSON path without downloading the real 4M-parameter
ImageNet weights.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend import dependencies
from backend.main import app
from backend.service import severity_from_probability
from pneumonia.inference import PneumoniaPredictor, save_metadata
from pneumonia.model import build_model
from pneumonia.preprocessing import preprocess_bytes

IMG_SIZE = (96, 96)


def _png_bytes(size=(256, 256), value: int = 120) -> bytes:
    rng = np.random.default_rng(0)
    array = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    array[:] = value  # deterministic flat-ish radiograph stand-in
    image = Image.fromarray(array, mode="RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def test_predictor(tmp_path_factory) -> PneumoniaPredictor:
    directory = tmp_path_factory.mktemp("model")
    model_path = directory / "tiny_model.keras"
    metadata_path = directory / "metadata.json"

    model = build_model(
        backbone="efficientnetb0",
        img_size=IMG_SIZE,
        weights=None,          # no download, weights are random
        augment=True,
        dense_units=0,
        learning_rate=1e-3,
    )
    model.save(model_path)

    save_metadata(
        {
            "model_name": "test_model",
            "backbone": "efficientnetb0",
            "img_size": list(IMG_SIZE),
            "class_names": ["NORMAL", "PNEUMONIA"],
            "positive_class": "PNEUMONIA",
            "threshold": 0.5,
            "created_at": "test",
        },
        metadata_path,
    )

    predictor = PneumoniaPredictor(model_path, metadata_path)
    assert predictor.is_ready, predictor.health()["error"]
    return predictor


@pytest.fixture()
def client(test_predictor, tmp_path, monkeypatch) -> TestClient:
    monkeypatch.setenv("CLINICAL_DATA_DIR", str(tmp_path / "clinical"))
    dependencies.set_predictor(test_predictor)
    with TestClient(app) as test_client:
        yield test_client
    dependencies.reset_predictor()


# --------------------------------------------------------------------------
# System endpoints
# --------------------------------------------------------------------------
def test_root_lists_endpoints(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "/api/predict" in response.json()["endpoints"]


def test_health_reports_ready_model(client: TestClient):
    payload = client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["model_ready"] is True
    assert payload["model"]["img_size"] == list(IMG_SIZE)


def test_model_endpoint(client: TestClient):
    payload = client.get("/api/model").json()
    assert payload["backbone"] == "efficientnetb0"
    assert payload["class_names"] == ["NORMAL", "PNEUMONIA"]


# --------------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------------
def test_predict_returns_contract(client: TestClient):
    response = client.post(
        "/api/predict",
        files={"file": ("xray.png", _png_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["prediction"] in {"NORMAL", "PNEUMONIA"}
    assert payload["is_pneumonia"] is (payload["prediction"] == "PNEUMONIA")
    assert 0.0 <= payload["confidence"] <= 1.0
    assert 0.0 <= payload["probability"] <= 1.0
    assert payload["probabilities"]["NORMAL"] + payload["probabilities"]["PNEUMONIA"] == pytest.approx(1.0, abs=1e-3)
    assert payload["threshold"] == 0.5
    assert payload["image"]["width"] == 256
    assert payload["processing_ms"] >= 0
    assert payload["severity"] in {"low", "moderate", "high"}
    assert "not a medical diagnosis" in payload["disclaimer"]


def test_predict_matches_shared_preprocessing(client: TestClient, test_predictor):
    """The endpoint must feed the model exactly what pneumonia.preprocessing returns."""
    data = _png_bytes()
    response = client.post("/api/predict", files={"file": ("xray.png", data, "image/png")})
    assert response.status_code == 200

    expected = test_predictor.probabilities(preprocess_bytes(data, IMG_SIZE)[np.newaxis, ...])[0]
    assert response.json()["probability"] == pytest.approx(float(expected), abs=1e-4)


def test_predict_rejects_non_image(client: TestClient):
    response = client.post(
        "/api/predict",
        files={"file": ("notes.txt", b"this is definitely not a radiograph", "text/plain")},
    )
    assert response.status_code == 400
    body = response.json()
    assert body["code"] in {"not_an_image", "unsupported_extension", "unsupported_media_type"}


def test_predict_rejects_empty_file(client: TestClient):
    response = client.post("/api/predict", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert response.json()["code"] == "empty_file"


def test_predict_rejects_oversized_file(client: TestClient, monkeypatch):
    monkeypatch.setattr("backend.validation.config.MAX_UPLOAD_BYTES", 1024)
    response = client.post(
        "/api/predict", files={"file": ("big.png", _png_bytes(size=(512, 512)), "image/png")}
    )
    assert response.status_code == 413
    assert response.json()["code"] == "file_too_large"


def test_predict_rejects_tiny_image(client: TestClient):
    response = client.post(
        "/api/predict", files={"file": ("tiny.png", _png_bytes(size=(16, 16)), "image/png")}
    )
    assert response.status_code == 400
    assert response.json()["code"] == "image_too_small"


def test_predict_requires_file_field(client: TestClient):
    response = client.post("/api/predict")
    assert response.status_code == 422


def test_authenticated_clinical_workflow_is_private_and_persisted(client: TestClient):
    cross_origin = client.post(
        "/api/auth/register",
        headers={"Origin": "https://untrusted.example"},
        json={
            "email": "blocked@example.test",
            "password": "secure-password-42",
            "display_name": "Blocked User",
        },
    )
    assert cross_origin.status_code == 403

    registration = client.post(
        "/api/auth/register",
        json={
            "email": "clinician@example.test",
            "password": "secure-password-42",
            "display_name": "Dr. Example",
        },
    )
    assert registration.status_code == 201, registration.text
    csrf_token = registration.json()["csrf_token"]
    headers = {"X-CSRF-Token": csrf_token}

    denied = client.post("/api/analyses", headers={}, files={"file": ("xray.png", _png_bytes(), "image/png")})
    assert denied.status_code == 403

    created = client.post(
        "/api/analyses",
        headers=headers,
        files={"file": ("xray.png", _png_bytes(), "image/png")},
        data={
            "patient_code": "PT-100",
            "patient_name": "Sample Patient",
            "age": "45",
            "sex": "Prefer not to say",
            "symptoms": "Cough",
        },
    )
    assert created.status_code == 201, created.text
    record = created.json()
    analysis_id = record["id"]
    explanation = record["result"]["explanation"]
    assert explanation["context_recorded"] is True
    assert explanation["clinical_context"]["used_for_prediction"] is False
    assert explanation["source"]["provider"] == "Local image classifier"
    assert explanation["source"]["gemini_used"] is False
    assert explanation["score_breakdown"]["normal_score"] + explanation["score_breakdown"]["pneumonia_score"] == pytest.approx(1.0, abs=1e-3)
    assert "Gemini" in record["result"]["privacy_notice"]
    assert "image_path" not in record

    dashboard = client.get("/api/dashboard")
    assert dashboard.status_code == 200
    assert dashboard.json()["analysis_count"] == 1
    assert dashboard.json()["patient_count"] == 1
    assert "image_path" not in dashboard.json()["recent_analyses"][0]

    patient_list = client.get("/api/patients").json()
    assert len(patient_list) == 1
    patient = client.get(f"/api/patients/{patient_list[0]['id']}").json()
    assert len(patient["analyses"]) == 1

    image = client.get(record["image_url"])
    assert image.status_code == 200
    assert image.headers["content-type"] == "image/png"
    assert image.headers["cache-control"] == "no-store"
    pdf = client.get(record["report_url"])
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.headers["cache-control"] == "no-store"
    assert pdf.content.startswith(b"%PDF-1.4")
    assert b"Gemini API used: No" in pdf.content

    other_client = TestClient(app)
    with other_client:
        other_registration = other_client.post(
            "/api/auth/register",
            json={
                "email": "another@example.test",
                "password": "another-password-42",
                "display_name": "Another User",
            },
        )
        assert other_registration.status_code == 201
        assert other_client.get(f"/api/analyses/{analysis_id}").status_code == 404
        assert other_client.get(record["image_url"]).status_code == 404

    patient_delete = client.delete(
        f"/api/patients/{patient_list[0]['id']}",
        headers=headers,
    )
    assert patient_delete.status_code == 204
    assert client.get(f"/api/analyses/{analysis_id}").status_code == 404


def test_password_change_invalidates_old_sessions(client: TestClient):
    registration = client.post(
        "/api/auth/register",
        json={
            "email": "password@example.test",
            "password": "initial-password-42",
            "display_name": "Password User",
        },
    )
    assert registration.status_code == 201
    old_session_cookie = client.cookies.get("clinical_session")

    changed = client.post(
        "/api/auth/password",
        headers={"X-CSRF-Token": registration.json()["csrf_token"]},
        json={"current_password": "initial-password-42", "new_password": "replacement-password-43"},
    )
    assert changed.status_code == 200, changed.text
    assert client.get("/api/auth/me").status_code == 200
    assert client.post(
        "/api/auth/login",
        json={"email": "password@example.test", "password": "initial-password-42"},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"email": "password@example.test", "password": "replacement-password-43"},
    ).status_code == 200

    stale_client = TestClient(app)
    stale_client.cookies.set("clinical_session", old_session_cookie)
    with stale_client:
        assert stale_client.get("/api/auth/me").status_code == 401


def test_pdf_report_paginates_long_clinical_notes():
    from backend.pdf_reports import build_analysis_pdf

    report = build_analysis_pdf(
        {
            "id": "analysis-id",
            "created_at": "2026-01-01T00:00:00+00:00",
            "patient_name": None,
            "patient_code": None,
            "age": None,
            "sex": None,
            "image_filename": "xray.png",
            "image_width": 256,
            "image_height": 256,
            "symptoms": "Detailed symptom note " * 100,
            "clinical_history": "",
            "notes": "",
            "result": {"prediction": "NORMAL", "confidence": 0.8, "probability": 0.2, "threshold": 0.5},
        }
    )
    assert report.startswith(b"%PDF-1.4")
    assert b"/Count 2" in report


def test_local_explanation_describes_threshold_without_inventing_findings():
    from backend.explanations import local_explanation

    explanation = local_explanation(
        {
            "prediction": "PNEUMONIA",
            "probability": 0.78,
            "confidence": 0.78,
            "threshold": 0.61,
            "probabilities": {"NORMAL": 0.22, "PNEUMONIA": 0.78},
            "model": {"name": "local-test", "backbone": "test-backbone", "img_size": [224, 224]},
        },
        symptoms="Cough",
    )
    assert explanation["decision_detail"]["threshold"] == 0.61
    assert "at or above" in explanation["decision_detail"]["explanation"]
    assert explanation["clinical_context"]["recorded"] is True
    assert explanation["clinical_context"]["used_for_prediction"] is False
    assert explanation["scope"]["image_findings_generated"] is False
    assert explanation["source"]["generative_ai_used"] is False


def test_predict_returns_503_without_model(tmp_path, monkeypatch):
    """A missing artifact must produce a clear 503, not a stack trace."""
    dependencies.set_predictor(
        PneumoniaPredictor(tmp_path / "missing.keras", tmp_path / "missing.json")
    )
    try:
        with TestClient(app) as test_client:  # lifespan must not crash
            assert test_client.get("/api/health").json()["model_ready"] is False
            response = test_client.post(
                "/api/predict", files={"file": ("xray.png", _png_bytes(), "image/png")}
            )
        assert response.status_code == 503
        assert response.json()["code"] == "model_unavailable"
    finally:
        dependencies.reset_predictor()


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "probability,expected",
    [(0.05, "low"), (0.5, "moderate"), (0.95, "high")],
)
def test_severity_banding(probability, expected):
    assert severity_from_probability(probability) == expected
