"""Reproducibility manifests: everything needed to re-run a version.

A manifest pins the dataset fingerprint, the run configuration, the
model hyper-parameters, the seeds and the runtime environment, and
carries a SHA-256 content hash so a tampered or drifted manifest is
detected instead of trusted. Two runs with the same manifest must
produce the same model.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, field


@dataclass
class ReproducibilityManifest:
    version_name: str
    dataset_fingerprint: str
    config: dict
    model_config: dict
    seeds: dict
    environment: dict = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self) -> None:
        if not self.environment:
            self.environment = {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "implementation": platform.python_implementation(),
            }
        if not self.content_hash:
            self.content_hash = self.compute_hash()

    def compute_hash(self) -> str:
        payload = json.dumps(
            {"version_name": self.version_name,
             "dataset_fingerprint": self.dataset_fingerprint,
             "config": self.config,
             "model_config": self.model_config,
             "seeds": self.seeds,
             "environment": self.environment},
            sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify(self) -> bool:
        """True when the manifest has not been altered since creation."""
        return self.compute_hash() == self.content_hash

    def to_dict(self) -> dict:
        return {"version_name": self.version_name,
                "dataset_fingerprint": self.dataset_fingerprint,
                "config": self.config,
                "model_config": self.model_config,
                "seeds": self.seeds,
                "environment": self.environment,
                "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, data: dict) -> "ReproducibilityManifest":
        return cls(version_name=data["version_name"],
                   dataset_fingerprint=data["dataset_fingerprint"],
                   config=data["config"],
                   model_config=data["model_config"],
                   seeds=data["seeds"],
                   environment=data.get("environment", {}),
                   content_hash=data["content_hash"])

    @classmethod
    def build(cls, version_name: str, dataset_fingerprint: str,
              config: dict, model_config: dict,
              seeds: dict) -> "ReproducibilityManifest":
        return cls(version_name=version_name,
                   dataset_fingerprint=dataset_fingerprint,
                   config=config, model_config=model_config, seeds=seeds)
