"""The candidate-scoring models compared in phase 2 (five algorithm families).

Every model is a scikit-learn Pipeline so that preprocessing learned from data
(imputation medians, scaling statistics) is fitted INSIDE each training fold
and never sees validation or student-test rows.

| key    | family                       | why it is a reasonable candidate                         |
|--------|------------------------------|----------------------------------------------------------|
| logreg | linear / probabilistic       | calibrated baseline; shows how far linear scoring goes   |
| knn    | instance-based               | non-parametric; tests whether similar pairs share labels |
| rf     | bagging ensemble of trees    | non-linear, robust to feature scale, little tuning       |
| hgb    | gradient boosting            | strong on tabular data, native NaN handling, fast        |
| mlp    | neural network               | smooth non-linear decision function                      |

Initial hyperparameters are deliberately ordinary defaults (documented below);
only the selected model is tuned (Bayesian optimisation, src/optimization).
"""
from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common.config import RANDOM_SEED

MODEL_INFO = {
    "logreg": {"name": "Logistic Regression", "family": "Linear (probabilistic)"},
    "knn": {"name": "k-Nearest Neighbours", "family": "Instance-based"},
    "rf": {"name": "Random Forest", "family": "Bagging ensemble (trees)"},
    "hgb": {"name": "HistGradientBoosting", "family": "Boosting ensemble (trees)"},
    "mlp": {"name": "MLP (neural network)", "family": "Neural network"},
}

DEFAULT_PARAMS: Dict[str, Dict[str, Any]] = {
    "logreg": {"C": 1.0, "max_iter": 1000},
    "knn": {"n_neighbors": 25, "weights": "distance"},
    "rf": {"n_estimators": 200, "min_samples_leaf": 5, "max_features": "sqrt", "max_depth": None},
    "hgb": {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31, "min_samples_leaf": 20,
            "l2_regularization": 0.0, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 20},
    "mlp": {"hidden_layer_sizes": (64, 32), "alpha": 1e-4, "learning_rate_init": 1e-3, "batch_size": 512,
            "max_iter": 60, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 8},
}

# What each model minimises (reported in the comparison table / report).
OBJECTIVES = {
    "logreg": "L2-regularised binary cross-entropy (log loss)",
    "knn": "none (lazy learner; distance-weighted vote of 25 neighbours)",
    "rf": "Gini impurity per split; bootstrap aggregation",
    "hgb": "binary cross-entropy (log loss), gradient boosting with early stopping",
    "mlp": "binary cross-entropy (log loss) + L2 penalty, Adam optimiser, early stopping",
}


def make_model(key: str, params: Dict[str, Any] | None = None) -> Pipeline:
    p = {**DEFAULT_PARAMS[key], **(params or {})}
    impute = ("impute", SimpleImputer(strategy="median", add_indicator=True))
    if key == "logreg":
        steps = [impute, ("scale", StandardScaler()), ("model", LogisticRegression(random_state=RANDOM_SEED, **p))]
    elif key == "knn":
        steps = [impute, ("scale", StandardScaler()), ("model", KNeighborsClassifier(n_jobs=-1, **p))]
    elif key == "rf":
        steps = [impute, ("model", RandomForestClassifier(n_jobs=-1, random_state=RANDOM_SEED, **p))]
    elif key == "hgb":
        # Native NaN support: no imputation (missing heading has its own split direction).
        steps = [("model", HistGradientBoostingClassifier(random_state=RANDOM_SEED, **p))]
    elif key == "mlp":
        steps = [impute, ("scale", StandardScaler()), ("model", MLPClassifier(random_state=RANDOM_SEED, **p))]
    else:
        raise KeyError(f"Unknown model '{key}'. Choose from {sorted(MODEL_INFO)}")
    return Pipeline(steps)


def training_history(pipeline: Pipeline) -> Dict[str, list]:
    """Epoch/iteration-wise histories, ONLY for models that genuinely record them."""
    model = pipeline.named_steps["model"]
    if isinstance(model, HistGradientBoostingClassifier) and getattr(model, "train_score_", None) is not None:
        # sklearn stores the NEGATIVE loss for the 'loss' scorer.
        return {"train_loss": (-model.train_score_).tolist(), "validation_loss": (-model.validation_score_).tolist()}
    if isinstance(model, MLPClassifier):
        hist = {"train_loss": list(model.loss_curve_)}
        if getattr(model, "validation_scores_", None) is not None:
            hist["validation_accuracy"] = list(model.validation_scores_)
        return hist
    return {}
