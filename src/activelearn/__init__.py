"""activelearn — an active-learning platform built from scratch.

From-scratch (pure stdlib) logistic regression and k-NN classifiers,
uncertainty/diversity query strategies, fail-closed labelling budgets,
a human-in-the-loop annotation queue with adjudication, retrain
triggers, learning-curve evaluation against a random baseline, a
gated model registry, reproducibility manifests and latency-budgeted
serving — all on seeded synthetic data.
"""

from activelearn.budget import BudgetExceeded, LabelingBudget
from activelearn.data import Dataset, LabelledPool, generate_dataset
from activelearn.evaluation import LearningCurve, compare_curves, run_curve
from activelearn.manifest import ReproducibilityManifest
from activelearn.models import (KNNClassifier, LogisticRegression,
                                MajorityBaseline, Standardizer)
from activelearn.pipeline import ActiveLearningRun, ModelVersion, RunConfig
from activelearn.queue import AnnotationQueue, cohens_kappa
from activelearn.registry import ModelRegistry, PromotionResult
from activelearn.serving import Prediction, ServingConfig, ServingLayer
from activelearn.strategies import STRATEGIES, select_batch
from activelearn.triggers import RetrainPolicy, RetrainState

__all__ = [
    "BudgetExceeded", "LabelingBudget",
    "Dataset", "LabelledPool", "generate_dataset",
    "LearningCurve", "compare_curves", "run_curve",
    "ReproducibilityManifest",
    "KNNClassifier", "LogisticRegression", "MajorityBaseline",
    "Standardizer",
    "ActiveLearningRun", "ModelVersion", "RunConfig",
    "AnnotationQueue", "cohens_kappa",
    "ModelRegistry", "PromotionResult",
    "Prediction", "ServingConfig", "ServingLayer",
    "STRATEGIES", "select_batch",
    "RetrainPolicy", "RetrainState",
]

__version__ = "1.0.0"
