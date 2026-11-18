"""Sudhe model: Random Forest.

    python scripts/models/sudhe_random_forest.py

Why: averages many deep trees grown on bootstrap samples with random feature subsets; low
variance, no scaling, a bagging counterpart to the boosting models.
Hyperparameters are Sudhe's; evaluation goes through the shared harness (36 features, fixed
trip folds, same Top-1 definition) so the result is comparable with every other model.
"""
from sklearn.ensemble import RandomForestClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"n_estimators": 120, "max_depth": 18, "min_samples_leaf": 2, "class_weight": "balanced"}


def build():
    return RandomForestClassifier(**CONFIG, random_state=SEED, n_jobs=-1)


if __name__ == "__main__":
    evaluate_model("sudhe_random_forest", "bagged trees (Random Forest)", build(), CONFIG)
