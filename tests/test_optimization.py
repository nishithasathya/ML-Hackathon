"""Bayesian optimisation: valid search spaces, recorded trials, persisted best parameters."""
import json

import optuna
import pytest

from src.models.cv import trip_folds
from src.models.dataset import build_partition
from src.models.model_zoo import make_model
from src.optimization.bayes_opt import SEARCH_SPACES, clean_params, make_objective, sample_params

optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.mark.parametrize("key", sorted(SEARCH_SPACES))
def test_sampled_parameters_build_valid_models(key):
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    for _ in range(5):
        trial = study.ask()
        params = clean_params(sample_params(key, trial))
        make_model(key, params)  # constructor accepts every sampled value
        study.tell(trial, 0.0)


def test_study_records_trials_and_best_params(multi_root, tmp_path):
    feats, obs = build_partition("train", data_root=multi_root, use_cache=False, verbose=False)
    folds = trip_folds(obs["trip_id"].unique().tolist(), n_splits=3)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42),
                                storage=f"sqlite:///{tmp_path / 'study.db'}", study_name="t")
    study.optimize(make_objective("hgb", feats, feats, obs, folds), n_trials=3)
    assert len(study.trials) == 3
    assert all("cv_macro_f1" in t.user_attrs and "params" in t.user_attrs for t in study.trials)
    best = clean_params(sample_params("hgb", optuna.trial.FixedTrial(study.best_params)))
    path = tmp_path / "best.json"
    path.write_text(json.dumps(best, default=str))
    assert json.loads(path.read_text())["max_iter"] == best["max_iter"]
    reloaded = optuna.load_study(study_name="t", storage=f"sqlite:///{tmp_path / 'study.db'}")
    assert len(reloaded.trials) == 3
