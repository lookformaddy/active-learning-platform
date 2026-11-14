"""Command-line interface: run a full active-learning experiment.

Usage:
    python -m activelearn.cli run --strategy margin --rounds 10 --batch 10
    python -m activelearn.cli compare --target 0.80
    python -m activelearn.cli serve --port 8000
"""

from __future__ import annotations

import argparse
import json
import sys

from activelearn.data import generate_dataset
from activelearn.evaluation import compare_curves, run_curve
from activelearn.manifest import ReproducibilityManifest
from activelearn.pipeline import ActiveLearningRun, RunConfig
from activelearn.registry import ModelRegistry


def _build_dataset(args) -> object:
    return generate_dataset(n_samples=args.n_samples, n_test=args.n_test,
                            seed=args.seed)


def cmd_run(args) -> int:
    dataset = _build_dataset(args)
    config = RunConfig(strategy=args.strategy, model_kind=args.model,
                       rounds=args.rounds, batch_size=args.batch,
                       total_budget=args.budget,
                       target_accuracy=args.target, seed=args.seed)
    run = ActiveLearningRun(dataset, config)
    version = run.run()
    manifest = ReproducibilityManifest.build(
        version.name, dataset.fingerprint(),
        {"strategy": config.strategy, "rounds": config.rounds,
         "batch_size": config.batch_size, "budget": config.total_budget,
         "target_accuracy": config.target_accuracy},
        {"model_kind": config.model_kind, "epochs": config.epochs},
        {"data": args.seed, "run": args.seed, "model": args.seed})
    version.manifest = manifest.to_dict()
    registry = ModelRegistry()
    registry.register(version, role="challenger")
    result = registry.promote_challenger()
    print(f"strategy={config.strategy} model={config.model_kind} "
          f"labels={version.n_labels} "
          f"test_accuracy={version.metrics['test_accuracy']:.4f} "
          f"baseline={version.baseline_metrics['majority_test_accuracy']:.4f}")
    print(f"labels to {config.target_accuracy:.2f} accuracy: "
          f"{version.metrics['labels_to_target']} (random: "
          f"{version.baseline_metrics['random_labels_to_target']})")
    print(f"promotion: {result.summary()}")
    print(f"manifest valid: {manifest.verify()} "
          f"({manifest.content_hash[:12]}...)")
    if args.json:
        print(json.dumps({"version": version.name,
                          "metrics": version.metrics,
                          "baseline_metrics": version.baseline_metrics,
                          "promotion": result.summary(),
                          "budget": run.budget.summary()},
                         indent=2))
    return 0


def cmd_compare(args) -> int:
    dataset = _build_dataset(args)
    active = run_curve(dataset, args.strategy, args.rounds, args.batch,
                       model_kind=args.model, seed=args.seed,
                       target_accuracy=args.target)
    control = run_curve(dataset, "random", args.rounds, args.batch,
                        model_kind=args.model, seed=args.seed,
                        target_accuracy=args.target)
    report = compare_curves(active, control)
    print(f"target accuracy: {args.target}")
    print(f"{args.strategy}: {report['active_labels_to_target']} labels "
          f"to target (final {report['active_final_accuracy']:.4f})")
    print(f"random: {report['random_labels_to_target']} labels to target "
          f"(final {report['random_final_accuracy']:.4f})")
    print(f"label savings: {report['label_savings_pct']}%")
    for point in active.points:
        print(f"  round {point.round_no:2d}  labels={point.n_labels:3d}  "
              f"acc={point.test_accuracy:.4f}")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from activelearn.service import app

    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="activelearn",
        description="Active-learning platform: query strategies, "
                    "labelling budgets and gated model promotion.")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p):
        p.add_argument("--strategy", default="margin",
                       choices=["least_confidence", "margin", "entropy",
                                "random", "diversity", "hybrid"])
        p.add_argument("--model", default="logistic",
                       choices=["logistic", "knn"])
        p.add_argument("--rounds", type=int, default=10)
        p.add_argument("--batch", type=int, default=10)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--n-samples", type=int, default=600)
        p.add_argument("--n-test", type=int, default=300)
        p.add_argument("--target", type=float, default=0.80)

    run_p = sub.add_parser("run", help="run one active-learning experiment")
    add_common(run_p)
    run_p.add_argument("--budget", type=int, default=100)
    run_p.add_argument("--json", action="store_true")
    run_p.set_defaults(func=cmd_run)

    cmp_p = sub.add_parser("compare",
                           help="learning curve: strategy vs random")
    add_common(cmp_p)
    cmp_p.set_defaults(func=cmd_compare)

    serve_p = sub.add_parser("serve", help="serve the champion over HTTP")
    serve_p.add_argument("--host", default="0.0.0.0")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.set_defaults(func=cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
