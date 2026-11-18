"""Model training: datasets, the five models, group CV, serialisation, selection rule."""
import joblib
import numpy as np
import pandas as pd
import pytest

from src.features.features import FEATURE_COLUMNS, LABEL_COLUMNS
from src.models.cv import cross_validate, trip_folds
from src.models.dataset import TrainingSubsetConfig, build_partition, training_subset, xy
from src.models.model_zoo import MODEL_INFO, make_model, training_history
from src.models.select_model import select


@pytest.fixture
def data(multi_root):
    return build_partition("train", data_root=multi_root, use_cache=False, verbose=False)


def test_dataset_keeps_string_ids_and_separates_labels(data):
    feats, obs = data
    assert feats["trip_id"].map(type).eq(str).all() and feats["trip_id"].str.len().eq(8).all()
    X, y = xy(feats)
    assert list(X.columns) == FEATURE_COLUMNS
    assert not set(LABEL_COLUMNS) & set(X.columns)
    assert set(np.unique(y)) == {0, 1}
    # obs keeps EVERY labelled fix, including the synthetic spike that has no candidate (1 per trip)
    assert len(obs) == feats.groupby(["trip_id", "obs_idx"]).ngroups + obs["trip_id"].nunique()
    assert obs["has_reference_candidate"].sum() <= feats.groupby(["trip_id", "obs_idx"]).ngroups


def test_training_subset_stride_and_offset_rule(data):
    feats, obs = data
    obs = obs.copy()
    obs.loc[obs["trip_id"] == "00000010", "ref_distance_m"] = 500.0  # pretend this trip is offset
    subset, info = training_subset(feats, obs, TrainingSubsetConfig(observation_stride=2))
    assert info["excluded_offset_trips"] == ["00000010"]
    assert "00000010" not in set(subset["trip_id"])
    assert (subset["obs_idx"] % 2 == 0).all()
    assert subset.groupby(["trip_id", "obs_idx"])["is_reference"].any().all()


def test_trip_folds_never_share_a_trip():
    folds = trip_folds([f"{i:08d}" for i in range(10)], n_splits=5)
    for train, val in folds:
        assert not set(train) & set(val)
    assert sorted(t for _, v in folds for t in v) == [f"{i:08d}" for i in range(10)]


@pytest.mark.parametrize("key", sorted(MODEL_INFO))
def test_each_model_trains_and_outputs_probabilities(data, key, tmp_path):
    feats, _ = data
    X, y = xy(feats)
    params = {"n_neighbors": 5} if key == "knn" else ({"n_estimators": 20} if key == "rf" else None)
    model = make_model(key, params)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    path = tmp_path / f"{key}.joblib"
    joblib.dump(model, path)
    assert np.allclose(joblib.load(path).predict_proba(X), proba)


def test_histories_only_for_models_that_record_them(data):
    feats, _ = data
    X, y = xy(feats)
    assert training_history(make_model("logreg").fit(X, y)) == {}
    assert "train_loss" in training_history(make_model("mlp", {"max_iter": 5}).fit(X, y))


def test_cross_validation_outputs_out_of_fold_scores(data):
    feats, obs = data
    folds = trip_folds(obs["trip_id"].unique().tolist(), n_splits=3)
    table, oof = cross_validate("logreg", None, feats, feats, obs, folds, verbose=False)
    assert len(table) == 3 and not np.isnan(oof).any()
    assert {"top1_accuracy", "macro_f1", "log_loss", "fit_time_s"} <= set(table.columns)


def test_selection_rule_prefers_better_probabilities_within_margin():
    comp = pd.DataFrame({"model": ["a", "b", "c"], "cv_top1_mean": [0.900, 0.897, 0.80], "cv_top1_std": [0.01] * 3,
                         "cv_macro_f1_mean": [0.9] * 3, "cv_log_loss_mean": [0.20, 0.10, 0.05],
                         "test_top1_accuracy": [0.9] * 3, "test_inference_time_s": [1, 1, 1]})
    decision = select(comp)
    assert decision["selected"] == "b" and decision["tied_within_margin"] == ["a", "b"]
