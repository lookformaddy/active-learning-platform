# Data

This project ships no data files. All data is synthetic and generated
in-process by `activelearn.data.generate_dataset` with a fixed seed:

- `shape="blobs"` (default): three Gaussian blobs in 6-D feature
  space — a near-linear problem, the logistic-regression showcase.
- `shape="moons"`: two interleaving crescents plus a central blob —
  a curved boundary where uncertainty sampling earns its keep.

The pool's hidden ground-truth labels are accessible only through
`LabelledPool.annotate`, which mirrors a human labelling exactly the
queried items. Nothing here comes from any real system or employer.
