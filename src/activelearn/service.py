"""FastAPI service for the active-learning platform.

Endpoints:
    GET  /health        — liveness + champion version
    POST /predict       — classify one feature vector (champion/canary/fallback)
    GET  /models        — registry versions, champion, challenger
    GET  /stats         — serving stats: arms, fallbacks, latency
    POST /promote       — promote the challenger if the registry gates pass

The module-level ``app`` is built at import with a champion trained on
the seeded synthetic data, so ``TestClient(app)`` works out of the box
in tests. For production wiring use :func:`build_app` with your own
ServingLayer.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from activelearn.data import generate_dataset, make_pool
from activelearn.models import LogisticRegression
from activelearn.pipeline import ModelVersion, _fit_and_score, RunConfig
from activelearn.registry import ModelRegistry
from activelearn.serving import ServingConfig, ServingLayer

__all__ = ["app", "build_app"]


class PredictRequest(BaseModel):
    features: list[float] = Field(description="Feature vector to classify")
    request_id: str | None = Field(default=None)


class PromoteResponse(BaseModel):
    promoted: bool
    summary: str
    checks: dict


def build_app(layer: ServingLayer) -> FastAPI:
    app = FastAPI(
        title="Active Learning Platform",
        description="From-scratch active learning: uncertainty and "
                    "diversity query strategies, fail-closed labelling "
                    "budgets, a human-in-the-loop annotation queue with "
                    "adjudication, retrain triggers, learning-curve "
                    "evaluation vs random sampling, and a gated model "
                    "registry with latency-budgeted serving.",
        version="1.0.0")

    @app.get("/health")
    def health() -> dict:
        registry = layer.registry
        return {"status": "ok",
                "champion": registry.champion_name,
                "challenger": registry.challenger_name}

    @app.post("/predict")
    def predict(req: PredictRequest) -> dict:
        if not req.features:
            raise HTTPException(status_code=422,
                                detail="features must not be empty")
        pred = layer.predict(req.features,
                             request_id=req.request_id or "api")
        return {"value": pred.value, "arm": pred.arm,
                "confidence": round(pred.confidence, 4),
                "latency_ms": round(pred.latency_ms, 2),
                "request_id": pred.request_id}

    @app.get("/models")
    def models() -> dict:
        return {"versions": layer.registry.list_versions(),
                "champion": layer.registry.champion_name,
                "challenger": layer.registry.challenger_name,
                "history": layer.registry.history}

    @app.get("/stats")
    def stats() -> dict:
        return layer.stats_dict()

    @app.post("/promote", response_model=PromoteResponse)
    def promote() -> PromoteResponse:
        try:
            result = layer.registry.promote_challenger()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return PromoteResponse(promoted=result.passed,
                               summary=result.summary(),
                               checks=result.checks)

    return app


def _default_layer() -> ServingLayer:
    """Seeded default: champion trained on the seed labels only."""
    dataset = generate_dataset(n_samples=300, n_test=200, seed=42)
    pool = make_pool(dataset, seed_per_class=3, seed=42)
    config = RunConfig(strategy="margin", rounds=1, batch_size=10,
                       epochs=200, seed=42)
    model, scaler, metrics = _fit_and_score(pool, dataset, config, 42)
    metrics["seed_accuracies"] = [metrics["test_accuracy"]]
    metrics["seed_accuracy_std"] = 0.0
    metrics["labels_to_target"] = None
    version = ModelVersion(
        name="v1-seed42", seed=42, strategy="margin",
        n_labels=pool.n_labeled, model_kind="logistic",
        metrics=metrics,
        baseline_metrics={"majority_test_accuracy": 0.3333,
                          "random_labels_to_target": None},
        model_state=model.to_dict() if isinstance(
            model, LogisticRegression) else {},
        scaler_state=scaler.to_dict())
    registry = ModelRegistry()
    registry.register(version, role="champion")
    return ServingLayer(registry, ServingConfig())


app = build_app(_default_layer())
