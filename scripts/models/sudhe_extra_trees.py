"""Sudhe model: Extra Trees.

    python scripts/models/sudhe_extra_trees.py

Why: like Random Forest but split thresholds are drawn at random, which lowers variance
further and trains faster.
Hyperparameters are Sudhe's; evaluation goes through the shared harness (36 features, fixed
trip folds, same Top-1 definition) so the result is comparable with every other model.
"""
from sklearn.ensemble import ExtraTreesClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 2, "class_weight": "balanced"}


def build():
    return ExtraTreesClassifier(**CONFIG, random_state=SEED, n_jobs=-1)


if __name__ == "__main__":
    evaluate_model("sudhe_extra_trees", "bagged trees (Extra Trees)", build(), CONFIG)
