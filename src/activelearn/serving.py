"""Latency-budgeted serving with a fail-closed fallback.

Every prediction goes through a latency budget: if the champion model
answers within budget, its prediction is returned; if it is slow, or
raises, the request falls back to the majority-class baseline instead
of failing the caller — but the fallback is recorded, never silent.
A canary fraction of traffic can be routed to the challenger, and
every request is accounted for in :meth:`stats_dict`.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from activelearn.models import MajorityBaseline
from activelearn.pipeline import ModelVersion


@dataclass
class Prediction:
    value: int
    arm: str  # champion | challenger | fallback
    confidence: float
    latency_ms: float
    request_id: str


@dataclass
class ServingConfig:
    latency_budget_ms: float = 100.0
    canary_pct: float = 0.0
    fallback_on_error: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.canary_pct <= 100.0:
            raise ValueError("canary_pct must be in [0, 100]")
        if self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be > 0")


class ServingLayer:
    """Serves a champion model with canary + latency-budget fallback."""

    def __init__(self, registry, config: ServingConfig | None = None,
                 predict_fn=None) -> None:
        self.registry = registry
        self.config = config or ServingConfig()
        self._predict_fn = predict_fn  # injectable for latency tests
        self.n_requests = 0
        self.n_fallback = 0
        self.n_canary = 0
        self.latencies_ms: list[float] = []
        self.fallback_reasons: list[str] = []

    def _route(self, request_id: str) -> str:
        if self.config.canary_pct <= 0:
            return "champion"
        digest = int(hashlib.sha256(request_id.encode()).hexdigest(), 16)
        return "challenger" if digest % 100 < self.config.canary_pct \
            else "champion"

    def _model_predict(self, version: ModelVersion,
                       features: list[float]) -> tuple[int, float]:
        if self._predict_fn is not None:
            return self._predict_fn(version, features)
        from activelearn.models import (KNNClassifier, LogisticRegression,
                                        Standardizer)
        scaler = Standardizer.from_dict(version.scaler_state)
        X = scaler.transform([features])
        if version.model_kind == "logistic":
            model = LogisticRegression.from_dict(version.model_state)
        elif version.model_kind == "knn":
            model = KNNClassifier.from_dict(version.model_state)
        else:
            raise ValueError(f"unknown model_kind {version.model_kind!r}")
        proba = model.predict_proba(X)[0]
        pred = max(range(len(proba)), key=lambda c: proba[c])
        return pred, proba[pred]

    def _fallback(self, request_id: str, latency_ms: float,
                  reason: str) -> Prediction:
        champion = self.registry.champion()
        baseline = MajorityBaseline()
        if champion is not None:
            # Baseline distribution comes from the champion's training
            # labels, recovered from its stored metrics if available.
            counts = champion.metrics.get("train_class_counts") or {}
            labels = []
            for k, v in counts.items():
                labels.extend([int(k)] * int(v))
            if labels:
                baseline.fit(labels)
        self.n_fallback += 1
        self.fallback_reasons.append(reason)
        return Prediction(value=baseline.majority, arm="fallback",
                          confidence=baseline.distribution.get(
                              baseline.majority, 0.0),
                          latency_ms=latency_ms, request_id=request_id)

    def predict(self, features: list[float],
                request_id: str) -> Prediction:
        self.n_requests += 1
        start = time.perf_counter()
        arm = self._route(request_id)
        version = (self.registry.champion() if arm == "champion"
                   else self.registry.versions.get(
                       self.registry.challenger_name or ""))
        if version is None:
            version = self.registry.champion()
            arm = "champion"
        if version is None:
            latency = (time.perf_counter() - start) * 1000.0
            self.latencies_ms.append(latency)
            return self._fallback(request_id, latency, "no_model")
        try:
            value, confidence = self._model_predict(version, features)
        except Exception as exc:  # noqa: BLE001 — serving must not crash
            latency = (time.perf_counter() - start) * 1000.0
            self.latencies_ms.append(latency)
            if self.config.fallback_on_error:
                return self._fallback(request_id, latency,
                                      f"model_error: {exc}")
            raise
        latency = (time.perf_counter() - start) * 1000.0
        self.latencies_ms.append(latency)
        if latency > self.config.latency_budget_ms:
            return self._fallback(request_id, latency,
                                  "latency_budget_exceeded")
        if arm == "challenger":
            self.n_canary += 1
        return Prediction(value=value, arm=arm, confidence=confidence,
                          latency_ms=latency, request_id=request_id)

    def stats_dict(self) -> dict:
        lat = sorted(self.latencies_ms)
        p50 = lat[len(lat) // 2] if lat else 0.0
        p95 = lat[int(len(lat) * 0.95)] if lat else 0.0
        return {"n_requests": self.n_requests,
                "n_fallback": self.n_fallback,
                "fallback_rate": round(
                    self.n_fallback / self.n_requests, 4)
                if self.n_requests else 0.0,
                "n_canary": self.n_canary,
                "p50_latency_ms": round(p50, 2),
                "p95_latency_ms": round(p95, 2)}
