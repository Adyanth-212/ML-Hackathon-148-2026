"""Advitha model 3: XGBoost (gradient-boosted trees, depth-wise growth).

    python scripts/models/advitha_xgb.py

Why: the third boosting implementation; depth-wise trees and second-order updates make it
a different inductive bias from LightGBM's leaf-wise trees. Objective: binary:logistic.
"""
from xgboost import XGBClassifier

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"objective": "binary:logistic", "n_estimators": 400, "learning_rate": 0.05, "max_depth": 8,
          "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
          "tree_method": "hist", "eval_metric": "logloss"}

def build():
    return XGBClassifier(**CONFIG, random_state=SEED, n_jobs=-1)


if __name__ == "__main__":
    evaluate_model("advitha_xgb", "gradient boosting (XGBoost)", build(), CONFIG)
