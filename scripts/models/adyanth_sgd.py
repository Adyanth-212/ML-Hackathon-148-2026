"""Adyanth model 3 (optional in SPLIT.md): linear classifier trained by stochastic gradient descent.

    python scripts/models/adyanth_sgd.py

Same model family as Logistic Regression (log loss, L2 penalty) but fitted by SGD, which
scales to far more rows than lbfgs. Shows whether the solver matters on our data.
Scaler inside the Pipeline (SGD is very sensitive to feature scale).
"""
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.harness import SEED, evaluate_model

CONFIG = {"loss": "log_loss", "penalty": "l2", "alpha": 1e-4, "class_weight": "balanced", "max_iter": 100,
          "tol": 1e-4, "learning_rate": "optimal", "scaler": "standard"}


def build():
    return Pipeline([("scale", StandardScaler()),
                     ("clf", SGDClassifier(loss=CONFIG["loss"], penalty=CONFIG["penalty"], alpha=CONFIG["alpha"],
                                           class_weight=CONFIG["class_weight"], max_iter=CONFIG["max_iter"],
                                           tol=CONFIG["tol"], learning_rate=CONFIG["learning_rate"],
                                           random_state=SEED))])


if __name__ == "__main__":
    evaluate_model("adyanth_sgd", "linear (SGD, log loss)", build(), CONFIG)
