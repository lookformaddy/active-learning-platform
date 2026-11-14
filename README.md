# Active Learning Platform

An active-learning platform built entirely from scratch (pure Python
stdlib — no sklearn, numpy or torch): uncertainty and diversity query
strategies over from-scratch logistic regression and k-NN classifiers,
fail-closed labelling budgets, a human-in-the-loop annotation queue
with adjudication, retrain triggers, learning-curve evaluation against
a random baseline, a gated model registry, reproducibility manifests
and latency-budgeted serving — all on seeded synthetic data.

## The enterprise problem

Labelling is the most expensive step in building a classifier, and
most labelling budgets are spent on examples the model already gets
right. Active learning fixes that by asking humans to label only the
examples the model is most uncertain about — but only if the loop is
governed: budgets must be hard caps, annotator disagreement must be
adjudicated rather than silently dropped, retraining must be triggered
by evidence rather than habit, and a new model must prove it beats
both the baseline and random sampling before it becomes the champion.
This project is that loop, end to end.

## Demo results (computed live, `scripts/demo.py`)

On the moons dataset (two interleaving crescents + a central blob,
800 pool / 1,000 test, mean of 3 seeds, target accuracy 0.78):

| Model | Margin sampling | Random sampling | Label savings |
|---|---|---|---|
| k-NN | reaches 0.78 with **126 labels** | never reaches 0.78 within the 206-label budget | margin reaches the target; random does not |
| Logistic regression | reaches 0.78 with **136 labels** | reaches 0.78 with 156 labels | **12.8%** |

Final accuracy at 206 labels: k-NN 0.792 (margin) vs 0.767 (random);
logistic 0.800 vs 0.778. A full governed run (margin/k-NN, 15 rounds
x 10 labels, budget 150) finishes at **0.793 test accuracy vs a 0.340
majority baseline**, with 16 retrains, 5 annotator disagreements all
adjudicated (Cohen's kappa 0.7418), and all four promotion gates
passing. Serving answers in ~0.4 ms; a 0.0001 ms budget forces the
recorded majority-baseline fallback.

## Architecture overview

```
seed labels ──► query strategy ──► labelling budget (fail-closed)
                     │                     │
                     ▼                     ▼
              annotation queue ──► adjudication (senior annotator)
                     │                     │
                     ▼                     ▼
              retrain triggers ──► retrain ──► learning-curve eval
                                                        │
                                                        ▼
                              versioned registry ──► gated promotion
                                                        │
                                                        ▼
                              latency-budgeted serving ──► fallback
```

See [`docs/architecture.md`](docs/architecture.md) for the full design,
including the module map and the per-component contracts.

### Modules (`src/activelearn/`)

| Module | Responsibility |
|---|---|
| `data.py` | Seeded synthetic data (`blobs` and `moons` shapes), the labelled/unlabelled pool, dataset fingerprinting |
| `models.py` | From-scratch `LogisticRegression`, `KNNClassifier`, `Standardizer`, `MajorityBaseline` |
| `strategies.py` | `least_confidence`, `margin`, `entropy`, `random`, `diversity` (cluster-based), `hybrid` |
| `budget.py` | `LabelingBudget` — total / per-round / cost caps, fail-closed, audited ledger |
| `queue.py` | `AnnotationQueue` — assignment, double-labelling, disagreement, adjudication, Cohen's kappa |
| `triggers.py` | `RetrainPolicy` — batch-size, uncertainty, drift and staleness triggers |
| `evaluation.py` | `run_curve`, `mean_learning_curve`, `compare_curves` (label savings) |
| `pipeline.py` | `ActiveLearningRun` — the full governed loop; emits a `ModelVersion` |
| `registry.py` | `ModelRegistry` — accuracy floor, must-beat-baseline, seed stability, label efficiency; rollback |
| `manifest.py` | `ReproducibilityManifest` — SHA-256-pinned config/data/seeds/environment |
| `serving.py` | `ServingLayer` — latency budget, canary routing, recorded majority-baseline fallback |
| `service.py` | FastAPI service (`/health`, `/predict`, `/models`, `/stats`, `/promote`) |
| `cli.py` | `run` / `compare` / `serve` commands |

## How to run it

```bash
pip install -r requirements.txt

# Full end-to-end demo (learning curves, governed run, budget refusal,
# promotion gates, serving fallback, manifest verification)
PYTHONPATH=src python scripts/demo.py

# One governed experiment (margin sampling, k-NN, 10 rounds x 10 labels)
PYTHONPATH=src python -m activelearn.cli run --strategy margin --model knn

# Learning curve: a strategy vs random sampling
PYTHONPATH=src python -m activelearn.cli compare --strategy margin --model logistic

# Serve the champion over HTTP
PYTHONPATH=src python -m activelearn.cli serve --port 8000

# Tests (71 passed + 1 skipped on system Python without FastAPI;
# 77 passed with FastAPI installed)
PYTHONPATH=src python -m pytest tests/ -q
```

## Key design decisions and trade-offs

- **No ML libraries.** Logistic regression and k-NN are implemented
  from scratch so the platform's behaviour is fully inspectable and
  the interview story is "I built the loop, not just called it".
- **Learning curves are averaged over seeds.** `labels_to_target`
  on a single seed is noisy — one lucky round can cross the target
  early — so the headline savings come from mean curves over 3
  seeds, and the registry's label-efficiency gate treats "the
  challenger reaches the target and random never does" as a pass.
- **Budgets are fail-closed, never partial.** A request over any cap
  raises `BudgetExceeded` and is recorded; the run stops instead of
  silently overspending. A partial grant would make spend
  unauditable.
- **Disagreement is signal, not noise.** Double-labelled items that
  disagree go to a senior adjudicator and stay in the record; the
  demo's kappa (0.74) is reported alongside accuracy, because a
  falling kappa means the annotation guidelines need retraining, not
  the model.
- **The pool's hidden labels are annotator-only.** Production code
  never reads `y_pool` except through `LabelledPool.annotate`, which
  mirrors a human labelling exactly the queried items — so the
  evaluation cannot leak.
- **Retraining is trigger-gated, not per-label.** Batch size is the
  workhorse trigger; uncertainty and drift thresholds cover the
  cases where the world changes faster than the label count grows.
- **Serving falls back, never fails the caller.** A slow or erroring
  champion returns the majority baseline and the fallback is
  recorded with its reason, so an outage shows up in the stats
  instead of as a silent accuracy drop.
- **CI ships under `docs/`.** The definition lives at
  `docs/github-actions-ci.yml`; move it to `.github/workflows/ci.yml`
  to activate (the push integration cannot write workflow files).

## What it demonstrates

- Active-learning query strategies (uncertainty, margin, entropy,
  diversity, hybrid) and when each one wins — margin sampling on a
  curved boundary, diversity when the pool is full of near-duplicates.
- Governed labelling: hard budgets, human-in-the-loop adjudication,
  inter-annotator agreement as a quality signal.
- Honest evaluation: learning curves vs a random control, multi-seed
  averaging, label savings as the headline metric.
- Production ML hygiene: gated champion promotion (accuracy floor,
  must-beat-baseline, multi-seed stability, label efficiency),
  reproducibility manifests with tamper detection, latency-budgeted
  serving with a recorded fallback.

## Clean-room note

All code is original and all data is synthetic, generated in-process
with fixed seeds. No employer's code, data, internal architecture,
project names or proprietary designs are included. Domains are generic.
