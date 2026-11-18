"""Sudhe model: Decision Tree.

    python scripts/models/sudhe_decision_tree.py

Why: one interpretable tree; the baseline that shows what the ensembles add.
Hyperparameters are Sudhe's; evaluation goes through the shared harness (36 features, fixed
trip folds, same Top-1 definition) so the result is comparable with every other model.
"""
from sklearn.tree import DecisionTreeClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 10, "class_weight": "balanced"}


def build():
    return DecisionTreeClassifier(**CONFIG, random_state=SEED)


if __name__ == "__main__":
    evaluate_model("sudhe_decision_tree", "single tree", build(), CONFIG)
