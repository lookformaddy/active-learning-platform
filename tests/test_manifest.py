"""Tests for the reproducibility manifest."""
from activelearn.manifest import ReproducibilityManifest


def _manifest(**overrides):
    data = {"version_name": "margin-knn-v1",
            "dataset_fingerprint": "abc123",
            "config": {"strategy": "margin", "rounds": 10},
            "model_config": {"model_kind": "knn"},
            "seeds": {"data": 42, "run": 42}}
    data.update(overrides)
    return ReproducibilityManifest.build(**data)


def test_manifest_verifies_when_untouched():
    manifest = _manifest()
    assert manifest.verify()
    assert len(manifest.content_hash) == 64


def test_manifest_detects_tampering():
    manifest = _manifest()
    tampered = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered.config["rounds"] = 999
    assert not tampered.verify()
    tampered2 = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered2.seeds["data"] = 7
    assert not tampered2.verify()


def test_manifest_roundtrip():
    manifest = _manifest()
    restored = ReproducibilityManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    assert restored.verify()
    assert restored.environment["python"]
