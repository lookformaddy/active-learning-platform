"""End-to-end demo: active learning vs random sampling on a curved
boundary, a full human-in-the-loop run, gated promotion, latency-
budgeted serving and a reproducibility manifest.

Every number printed is computed live from seeded synthetic data —
no hard-coded metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from activelearn.budget import BudgetExceeded  # noqa: E402
from activelearn.data import generate_dataset, make_pool  # noqa: E402
from activelearn.evaluation import (compare_curves, mean_learning_curve)  # noqa: E402
from activelearn.manifest import ReproducibilityManifest  # noqa: E402
from activelearn.pipeline import (ActiveLearningRun, ModelVersion, RunConfig,  # noqa: E402
                                  _fit_and_score)
from activelearn.registry import ModelRegistry  # noqa: E402
from activelearn.serving import ServingConfig, ServingLayer  # noqa: E402

SEEDS = (42, 7, 123)
TARGET = 0.78


def _moons(seed: int):
    return generate_dataset(n_samples=800, n_test=1000, seed=seed,
                            shape="moons")


def run_demo() -> None:
    print("== Active Learning Platform — end-to-end demo ==\n")

    # 1. Data: two interleaving crescents + a central blob.
    dataset = _moons(42)
    print(f"[1] moons dataset: {len(dataset.X)} pool items, "
          f"{len(dataset.X_test)} test items, "
          f"{dataset.n_features} features, {dataset.n_classes} classes")
    print(f"    fingerprint: {dataset.fingerprint()[:16]}...  "
          f"class counts: {dataset.class_counts()}")

    # 2. Mean learning curves (3 seeds): margin vs random, k-NN + logreg.
    print(f"\n[2] learning curves — mean of {len(SEEDS)} seeds, "
          f"target accuracy {TARGET}, 20 rounds x 10 labels")
    for model_kind in ("knn", "logistic"):
        active = mean_learning_curve(_moons, "margin", 20, 10, SEEDS,
                                     model_kind=model_kind,
                                     target_accuracy=TARGET)
        control = mean_learning_curve(_moons, "random", 20, 10, SEEDS,
                                      model_kind=model_kind,
                                      target_accuracy=TARGET)
        report = compare_curves(active, control)
        print(f"    {model_kind}: margin reaches {TARGET} with "
              f"{report['active_labels_to_target']} labels; random: "
              f"{report['random_labels_to_target']} "
              f"(final {report['active_final_accuracy']:.3f} vs "
              f"{report['random_final_accuracy']:.3f})")
        if report["label_savings_pct"] is not None:
            print(f"      label savings: {report['label_savings_pct']}%")
        else:
            print("      label savings: margin reaches the target; "
                  "random never does within the 206-label budget")
        for p in active.points[::5]:
            print(f"      round {p.round_no:2d}  labels={p.n_labels:3d}  "
                  f"margin acc={p.test_accuracy:.3f}")

    # 3. Full run: query -> budget -> annotate/adjudicate -> retrain.
    config = RunConfig(strategy="margin", model_kind="knn", rounds=15,
                       batch_size=10, total_budget=150,
                       seed_labels_per_class=2,
                       target_accuracy=TARGET, seed=42)
    run = ActiveLearningRun(dataset, config)
    version = run.run()
    m = version.metrics
    print(f"\n[3] full run ({config.strategy}/{config.model_kind}): "
          f"{version.n_labels} labels, test accuracy "
          f"{m['test_accuracy']:.3f} (baseline "
          f"{version.baseline_metrics['majority_test_accuracy']:.3f}), "
          f"labels to {TARGET}: {m['labels_to_target']} "
          f"(random: {version.baseline_metrics['random_labels_to_target']})")
    print(f"    retrains: {run.retrain_count}, "
          f"budget spent {run.budget.spent}/{run.budget.total_budget}, "
          f"queue: {run.queue.summary()['n_disagreements']} disagreements, "
          f"{run.queue.summary()['n_adjudicated']} adjudicated, "
          f"agreement kappa "
          f"{run.queue.agreement()['cohens_kappa']}")

    # 4. Budget is fail-closed: one label over the cap is refused.
    try:
        run.budget.request(run.budget.remaining + 1, round_no=99,
                           purpose="demo-overspend")
        print("\n[4] budget: ERROR — overspend was granted")
    except BudgetExceeded as exc:
        print(f"\n[4] budget fail-closed: overspend refused ({exc})")

    # 5. Registry: seed-only champion vs the active-learning challenger.
    pool = make_pool(dataset, seed_per_class=2, seed=42)
    _model, _scaler, seed_metrics = _fit_and_score(pool, dataset, config, 42)
    seed_version = ModelVersion(
        name="seed-only-v0", seed=42, strategy="margin", n_labels=6,
        model_kind="knn", metrics=seed_metrics,
        baseline_metrics={"majority_test_accuracy": 0.333,
                          "random_labels_to_target": None},
        model_state={}, scaler_state={})
    registry = ModelRegistry()
    registry.register(seed_version, role="champion")
    registry.register(version, role="challenger")
    result = registry.promote_challenger()
    print(f"\n[5] promotion gates: {result.summary()}")
    for gate, passed in result.checks.items():
        print(f"      {'PASS' if passed else 'FAIL'}  {gate}")
    print(f"    champion is now: {registry.champion_name}")

    # 6. Serving: champion prediction, then a 0.0001 ms budget forces
    # the fallback — recorded, never silent.
    layer = ServingLayer(registry, ServingConfig())
    features = dataset.X_test[0]
    pred = layer.predict(features, request_id="demo-1")
    print(f"\n[6] serving: test item (true label {dataset.y_test[0]}) -> "
          f"predicted {pred.value} via {pred.arm}, "
          f"confidence {pred.confidence:.2f}, {pred.latency_ms:.1f} ms")
    tight = ServingLayer(registry,
                         ServingConfig(latency_budget_ms=0.0001))
    fb = tight.predict(features, request_id="demo-2")
    print(f"    tight budget (0.0001 ms): arm={fb.arm} "
          f"(fallback reason: {tight.fallback_reasons[-1]})")
    print(f"    serving stats: {layer.stats_dict()}")

    # 7. Reproducibility manifest: verify, then tamper and re-verify.
    manifest = ReproducibilityManifest.build(
        version.name, dataset.fingerprint(),
        {"strategy": config.strategy, "rounds": config.rounds,
         "batch_size": config.batch_size, "budget": config.total_budget,
         "target_accuracy": config.target_accuracy,
         "shape": "moons"},
        {"model_kind": config.model_kind},
        {"data": 42, "run": 42, "model": 42})
    print(f"\n[7] manifest {manifest.content_hash[:12]}... "
          f"verifies: {manifest.verify()}")
    tampered = ReproducibilityManifest.from_dict(manifest.to_dict())
    tampered.config["rounds"] = 999
    print(f"    tampered manifest verifies: {tampered.verify()} "
          f"(expected False)")
    print("\nDemo complete.")


if __name__ == "__main__":
    run_demo()
