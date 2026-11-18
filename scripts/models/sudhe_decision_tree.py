"""Sudhe model: Decision Tree."""
from sklearn.tree import DecisionTreeClassifier
from _sudhe_models import run_model

CONFIG = {"max_depth": 8, "min_samples_split": 20, "min_samples_leaf": 10, "class_weight": "balanced", "random_state": 148}
if __name__ == "__main__":
    run_model("sudhe_decision_tree", DecisionTreeClassifier(**CONFIG), CONFIG)
