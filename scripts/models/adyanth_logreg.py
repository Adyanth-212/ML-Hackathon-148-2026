"""Adyanth model 1: Logistic Regression (linear, probabilistic).

    python scripts/models/adyanth_logreg.py

Scaler inside the Pipeline so each CV fold refits it. class_weight='balanced' because there is
about one positive per 12 candidates. Pre-harness prototype: see _adyanth_proto.py.
"""
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _adyanth_proto import SEED, evaluate  # noqa: E402

CONFIG = {"C": 1.0, "penalty": "l2", "class_weight": "balanced", "solver": "lbfgs", "max_iter": 1000, "scaler": "standard"}

if __name__ == "__main__":
    est = Pipeline([("scale", StandardScaler()),
                    ("clf", LogisticRegression(C=CONFIG["C"], class_weight=CONFIG["class_weight"], solver=CONFIG["solver"],
                                               max_iter=CONFIG["max_iter"], random_state=SEED))])
    evaluate("adyanth_logreg", "linear / probabilistic", est, CONFIG)
