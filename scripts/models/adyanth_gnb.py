"""Adyanth model 2: Gaussian Naive Bayes (generative, probabilistic).

    python scripts/models/adyanth_gnb.py

Assumes the features are independent and Gaussian given the class. Ours are strongly
correlated (several distance measures, prev/next heading), so this model shows what the
independence assumption costs, in calibration (log loss) above all. The scaler does not
change GNB; it is kept in the Pipeline for a uniform interface.
"""
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pathsnap.harness import evaluate_model

CONFIG = {"var_smoothing": 1e-9, "priors": "empirical", "scaler": "standard"}

if __name__ == "__main__":
    est = Pipeline([("scale", StandardScaler()), ("clf", GaussianNB(var_smoothing=CONFIG["var_smoothing"]))])
    evaluate_model("adyanth_gnb", "probabilistic (generative)", est, CONFIG)
