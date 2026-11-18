"""Advitha model 2: scikit-learn HistGradientBoostingClassifier.

    python scripts/models/advitha_hgb.py

Why: the scikit-learn implementation of histogram gradient boosting (LightGBM-style);
a useful check that the boosting result is not library-specific. Objective: binary log
loss. Early stopping is off so every fold trains the same number of trees.
"""
from sklearn.ensemble import HistGradientBoostingClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"loss": "log_loss", "max_iter": 300, "learning_rate": 0.1, "max_leaf_nodes": 63,
          "min_samples_leaf": 50, "l2_regularization": 1.0, "early_stopping": False}

if __name__ == "__main__":
    est = HistGradientBoostingClassifier(**CONFIG, random_state=SEED)
    evaluate_model("advitha_hgb", "gradient boosting (sklearn HistGB)", est, CONFIG)
