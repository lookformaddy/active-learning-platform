"""API tests (FastAPI TestClient). These run only when fastapi and
httpx are installed; otherwise they are skipped, not failed.
"""
import pytest

fastapi = pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402

from activelearn.service import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["champion"] == "v1-seed42"


def test_predict():
    resp = client.post("/predict", json={
        "features": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "request_id": "api-1",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["value"] in (0, 1, 2)
    assert 0.0 < data["confidence"] <= 1.0
    assert data["arm"] in ("champion", "challenger", "fallback")


def test_predict_empty_features_rejected():
    resp = client.post("/predict", json={"features": []})
    assert resp.status_code == 422


def test_models_endpoint():
    resp = client.get("/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["champion"] == "v1-seed42"
    assert any(v["name"] == "v1-seed42" for v in data["versions"])


def test_stats_endpoint():
    client.post("/predict", json={"features": [0.1] * 6})
    resp = client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_requests"] >= 1
    assert "p95_latency_ms" in data


def test_promote_without_challenger_rejected():
    resp = client.post("/promote")
    assert resp.status_code == 400
