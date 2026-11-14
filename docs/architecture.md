# Architecture — Active Learning Platform

## Overview

The platform is a governed loop around one scarce resource (human
labels) and one decision (which model serves). Data flows in one
direction — pool → strategy → budget → queue → triggers → retrain →
evaluation → registry → serving — and every step that spends money or
changes what serves is gated and recorded.

```
                    ┌──────────────────────────────┐
                    │  Synthetic data (seeded)     │
                    │  blobs: near-linear          │
                    │  moons: curved boundary      │
                    └──────────────┬───────────────┘
                                   ▼
              ┌─────────────────────────────────────────┐
              │ LabelledPool: labelled set + pool       │
              │ (hidden truth reachable only via        │
              │  annotate() — the annotator contract)   │
              └──────────────┬──────────────────────────┘
                             ▼
        ┌──────────────────────────────────────────────────┐
        │ Query strategies (strategies.py)                  │
        │ least-confidence · margin · entropy · random      │
        │ diversity (k-means) · hybrid                      │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Labelling budget (budget.py) — FAIL-CLOSED        │
        │ total / per-round / cost caps; refused requests   │
        │ raise BudgetExceeded and are written to a ledger  │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Annotation queue (queue.py)                       │
        │ assign → label → double-label → disagree →        │
        │ senior adjudication; Cohen's kappa per run        │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Retrain triggers (triggers.py)                    │
        │ batch size · mean pool entropy · centroid drift   │
        │ · staleness (max rounds between retrains)         │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Models (models.py) — from scratch, pure stdlib    │
        │ LogisticRegression (softmax GD) · KNN (distance-  │
        │ weighted voting) · Standardizer · MajorityBaseline│
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Evaluation (evaluation.py)                        │
        │ learning curves vs a random control, averaged     │
        │ over seeds; labels-to-target + label savings      │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Registry (registry.py) — gated promotion          │
        │ accuracy floor · must-beat-baseline · seed         │
        │ stability · label efficiency; audited history,    │
        │ rollback to the previous champion                 │
        └──────────────┬───────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────┐
        │ Serving (serving.py)                              │
        │ latency budget + canary routing + recorded         │
        │ majority-baseline fallback; p50/p95 stats         │
        └──────────────────────────────────────────────────┘

  Cross-cutting: ReproducibilityManifest (manifest.py) pins the
  dataset fingerprint, run config, model config, seeds and runtime,
  with a SHA-256 content hash — tampering is detected, not trusted.
  FastAPI service (service.py) exposes the champion; CLI (cli.py)
  exposes run / compare / serve.
```

## Module contracts

### `data.py` — synthetic data and the pool
- `generate_dataset(..., shape="blobs" | "moons")` — deterministic
  for a fixed seed; the test split uses a separate RNG stream.
- `Dataset.fingerprint()` — SHA-256 over the data, recorded in
  manifests so a run can be tied to its exact data.
- `LabelledPool` — the single most important invariant: `y_pool`
  (hidden ground truth) is reachable only through `annotate()`,
  which mirrors a human labelling exactly the queried items. Test
  labels are never passed to the pool.
- `seed(per_class)` — stratified seed set, so every class is
  represented from round zero.

### `models.py` — from-scratch classifiers
- `LogisticRegression` — multinomial softmax, full-batch gradient
  descent, L2, stable softmax, `to_dict`/`from_dict` serialisation.
- `KNNClassifier` — inverse-distance-weighted voting returning a
  proper probability distribution, so every query strategy works on
  either model identically.
- `Standardizer` — fitted on the labelled set only; pool statistics
  can never leak into training.
- `MajorityBaseline` — the trivial classifier every learned model
  must beat, and the serving fallback.

### `strategies.py` — query strategies
All strategies return exactly `batch_size` indices, most valuable
first, with deterministic tie-breaking by pool index:
- **least_confidence** — lowest max-class probability.
- **margin** — smallest gap between the top two probabilities.
- **entropy** — highest predictive entropy.
- **random** — uniform sample; the control arm, and the baseline
  every strategy must beat.
- **diversity** — k-means over pool features, most uncertain item
  per cluster, so a batch never spends its budget on near-duplicates
  from one ambiguous region.
- **hybrid** — half the batch by margin, half by diversity.
- Errors are fail-closed: unknown strategy, batch larger than the
  pool, or an empty pool all raise `ValueError`.

### `budget.py` — fail-closed labelling budget
- `LabelingBudget.request(n, round_no)` grants the full batch or
  raises `BudgetExceeded` — partial grants never happen.
- Check order: per-round cap (a configuration bug), total budget,
  cost cap. Every refusal is written to the ledger with its reason,
  so spend is fully auditable.

### `queue.py` — human-in-the-loop annotation
- Items are assigned round-robin; an annotator never receives the
  same item twice.
- A configurable share of items is double-labelled. Agreement scores
  (percent agreement, Cohen's kappa) are computed over those items.
- Disagreement goes to a senior adjudicator whose label is final;
  the disagreement stays in the record — it marks an ambiguous
  region, it is not deleted.

### `triggers.py` — retrain triggers
Any one trigger fires a retrain, and the reason is recorded:
- **batch size** — enough new labels since the last retrain;
- **uncertainty** — mean pool entropy above a threshold;
- **drift** — centroid of newly labelled items shifted away from
  the training centroid;
- **staleness** — maximum rounds since the last retrain.

### `evaluation.py` — learning-curve evaluation
- `run_curve` runs one arm (strategy + model + seed) and records
  test accuracy after every round.
- `mean_learning_curve` averages per-round accuracy over several
  seeds — the headline numbers come from these means, because
  `labels_to_target` on a single seed is noisy.
- `compare_curves` reports `labels_to_target` for each arm and the
  derived `label_savings_pct`. When the active arm reaches the
  target and random never does within the budget, that is reported
  as such rather than as a fabricated percentage.

### `pipeline.py` — the governed run
`ActiveLearningRun` owns one training run: seed → query → budget →
queue (assign, label, double-label, adjudicate) → triggers → retrain
→ final fit → multi-seed stability check → learning-curve comparison
→ `ModelVersion`. The version carries everything the registry gates
need: test accuracy, per-class recall, seed accuracies, labels to
target, baseline metrics, and the serialised model + scaler.

### `registry.py` — gated champion promotion
Four gates, all must pass, every check recorded:
1. **accuracy_floor** — test accuracy at or above the floor;
2. **must_beat_baseline** — beats the majority baseline by at least
   the configured delta;
3. **seed_stability** — std of test accuracy across re-training
   seeds at or below the cap;
4. **label_efficiency** — reaches the target accuracy with no more
   labels than the random baseline needed (reaching a target random
   never reaches counts as a pass).
Promotion history is audited; `rollback()` restores the most recent
previously-promoted champion.

### `manifest.py` — reproducibility
`ReproducibilityManifest` pins version name, dataset fingerprint,
run config, model config, seeds and runtime environment under a
SHA-256 content hash. `verify()` detects any post-hoc edit.

### `serving.py` — latency-budgeted serving
- Canary routing by deterministic hash of the request id.
- A prediction slower than the latency budget, or a model error,
  returns the majority-baseline fallback — and the fallback is
  recorded with its reason, never silent.
- `stats_dict()` reports request counts, fallback rate and p50/p95
  latency.

### `service.py` / `cli.py`
- FastAPI: `GET /health`, `POST /predict`, `GET /models`,
  `GET /stats`, `POST /promote`. The module-level app ships with a
  champion trained on the seeded data so `TestClient(app)` works
  out of the box.
- CLI: `run` (one governed experiment), `compare` (strategy vs
  random learning curve), `serve` (HTTP serving).

## Why the moons dataset is the headline

On Gaussian blobs with a near-linear boundary, random sampling is
nearly optimal — a logistic model learns the boundary from a handful
of points, so there is no label-efficiency story to tell. The moons
shape (curved boundary, noise dimensions) is where uncertainty
sampling earns its keep: margin sampling reaches 0.78 accuracy with
126 labels (k-NN, mean of 3 seeds) while random sampling never gets
there inside the same 206-label budget. The project keeps both
shapes so the contrast itself is part of the demonstration: the right
question is not "does active learning work?" but "on which geometry
does it pay, and by how much?"

## Failure modes handled

| Failure | Handling |
|---|---|
| Labelling overspend | `BudgetExceeded`, run stops, refusal logged |
| Annotator disagreement | Senior adjudication; disagreement kept in the record |
| Annotator pool exhausted | `assign` raises when no unassigned annotator remains |
| Model too slow / errors at serve time | Recorded majority-baseline fallback |
| Challenger weaker than champion | Promotion gates reject; champion unchanged |
| Non-reproducible run | Manifest hash verification fails on tampering |
| Test-label leakage | Hidden pool labels reachable only via `annotate()`; test split on a separate RNG stream |
