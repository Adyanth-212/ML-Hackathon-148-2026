"""Advitha: LightGBM with the hyperparameters found by Bayesian optimisation (scripts/optimize.py).

    python scripts/optimize.py                     # first: the Optuna search on train CV
    python scripts/models/advitha_lgbm_tuned.py    # then: the harness (CV + student_test) with the best params

Same model, features, folds and seed as advitha_lgbm.py; only the hyperparameters differ, so
advitha_lgbm vs advitha_lgbm_tuned is the untuned vs tuned comparison (scripts/compare_tuning.py).
"""
import json

from lightgbm import LGBMClassifier

from pathsnap.config import repo_path
from pathsnap.harness import SEED, evaluate_model

_BEST = json.loads(repo_path("experiments/results/bayes_opt/best_params.json").read_text())
CONFIG = _BEST["params"]


def build():
    return LGBMClassifier(**CONFIG, random_state=SEED, n_jobs=-1, verbose=-1)


if __name__ == "__main__":
    evaluate_model("advitha_lgbm_tuned", "gradient boosting (LightGBM, Bayesian-tuned)", build(),
                   {**CONFIG, "tuned_by": f"optuna study {_BEST['study']}, trial {_BEST['best_trial']}"})
