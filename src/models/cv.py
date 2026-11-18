"""Group-aware cross-validation and evaluation helpers.

Folds are split by trip id (``GroupKFold``), so no trip ever contributes rows to
both the training and the validation side of a fold. Within a fold:

* the model is FITTED on the training-subset rows of the training trips
  (see ``dataset.training_subset``);
* it is EVALUATED on every candidate row of the held-out trips, so the fold
  metrics describe the real task, not the subsample.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src.evaluation.metrics import classification_metrics, observation_accuracy, select_top1
from src.models.dataset import xy
from src.models.model_zoo import make_model

N_SPLITS = 5


def trip_folds(trip_ids: List[str], n_splits: int = N_SPLITS) -> List[Tuple[List[str], List[str]]]:
    """Deterministic trip-level folds (GroupKFold on the sorted trip list)."""
    trips = np.array(sorted(map(str, trip_ids)))
    folds = []
    for tr, va in GroupKFold(n_splits=n_splits).split(trips, groups=trips):
        folds.append((trips[tr].tolist(), trips[va].tolist()))
    return folds


def score_candidates(model, feats: pd.DataFrame) -> Tuple[np.ndarray, float]:
    X, _ = xy(feats)
    start = time.perf_counter()
    proba = model.predict_proba(X)[:, 1]
    return proba, time.perf_counter() - start


def evaluate_scores(feats: pd.DataFrame, obs: pd.DataFrame, proba: np.ndarray) -> Dict[str, float]:
    """All candidate-level and observation-level metrics for one scored table."""
    out = classification_metrics(feats["is_reference"].to_numpy(dtype=int), proba)
    scored = feats[["trip_id", "obs_idx", "arc_id", "distance_rank", "is_reference"]].assign(score=proba)
    out.update(observation_accuracy(select_top1(scored), obs))
    return out


def cross_validate(key: str, params: Optional[Dict[str, Any]], fit_rows: pd.DataFrame, all_rows: pd.DataFrame,
                   obs: pd.DataFrame, folds, verbose: bool = True) -> Tuple[pd.DataFrame, np.ndarray]:
    """Return (per-fold metrics, out-of-fold probabilities aligned with ``all_rows``)."""
    oof = np.full(len(all_rows), np.nan)
    records = []
    for k, (train_trips, val_trips) in enumerate(folds):
        tr = fit_rows[fit_rows["trip_id"].isin(train_trips)]
        va_mask = all_rows["trip_id"].isin(val_trips).to_numpy()
        va = all_rows.loc[va_mask]
        model = make_model(key, params)
        X, y = xy(tr)
        start = time.perf_counter()
        model.fit(X, y)
        fit_s = time.perf_counter() - start
        proba, pred_s = score_candidates(model, va)
        oof[va_mask] = proba
        metrics = evaluate_scores(va, obs[obs["trip_id"].isin(val_trips)], proba)
        metrics.update({"model": key, "fold": k, "fit_time_s": fit_s, "predict_time_s": pred_s,
                        "n_fit_rows": int(len(tr)), "n_val_rows": int(len(va)), "val_trips": " ".join(val_trips)})
        records.append(metrics)
        if verbose:
            print(f"    {key} fold {k}: top1={metrics['top1_accuracy']:.4f} macroF1={metrics['macro_f1']:.4f} "
                  f"logloss={metrics['log_loss']:.4f} fit={fit_s:.1f}s pred={pred_s:.1f}s", flush=True)
    return pd.DataFrame(records), oof
