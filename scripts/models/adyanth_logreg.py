"""Adyanth model 1: Logistic Regression (linear, probabilistic).

    python scripts/models/adyanth_logreg.py

Scaler inside the Pipeline, so each CV fold refits it. class_weight='balanced' because
about 1 in 11 candidates is positive. Also writes two feature-group ablations
(distance only; distance + direction) to show what each group adds.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.features import FEATURE_GROUPS
from pathsnap.harness import SEED, evaluate_model

CONFIG = {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 2000,
          "scaler": "standard"}


def build():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=CONFIG["C"], class_weight=CONFIG["class_weight"],
                                                solver=CONFIG["solver"], max_iter=CONFIG["max_iter"],
                                                random_state=SEED))])


if __name__ == "__main__":
    evaluate_model("adyanth_logreg", "linear / probabilistic", build(), CONFIG)
    dist = FEATURE_GROUPS["distance"] + FEATURE_GROUPS["density"]
    evaluate_model("ablation_logreg_distance", "linear / probabilistic (ablation)", build(),
                   {**CONFIG, "feature_groups": ["distance", "density"]}, features=dist)
    evaluate_model("ablation_logreg_distance_direction", "linear / probabilistic (ablation)", build(),
                   {**CONFIG, "feature_groups": ["distance", "density", "direction"]},
                   features=dist + FEATURE_GROUPS["direction"])
