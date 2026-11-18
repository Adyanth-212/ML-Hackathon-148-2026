"""Sudhe model: Extra Trees."""
from sklearn.ensemble import ExtraTreesClassifier
from _sudhe_models import run_model

CONFIG = {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 2, "class_weight": "balanced", "random_state": 148, "n_jobs": -1}
if __name__ == "__main__":
    run_model("sudhe_extra_trees", ExtraTreesClassifier(**CONFIG), CONFIG)
