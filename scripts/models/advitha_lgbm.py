"""Advitha model 1: LightGBM (gradient-boosted trees, leaf-wise growth).

    python scripts/models/advitha_lgbm.py

Why: strong on tabular features with mixed scales and interactions (distance x heading x
context), fast with histogram binning, no scaling needed. Objective: binary log loss
(probabilities feed the HMM emissions). No class weighting: Top-1 only needs the ranking
within an observation, and unweighted log loss keeps the probabilities calibrated.
"""
from lightgbm import LGBMClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"objective": "binary", "n_estimators": 400, "learning_rate": 0.05, "num_leaves": 63,
          "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
          "reg_lambda": 1.0}

def build():
    return LGBMClassifier(**CONFIG, random_state=SEED, n_jobs=-1, verbose=-1)


if __name__ == "__main__":
    evaluate_model("advitha_lgbm", "gradient boosting (LightGBM)", build(), CONFIG)
