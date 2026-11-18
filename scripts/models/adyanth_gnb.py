"""Adyanth model 2: Gaussian Naive Bayes (generative, probabilistic).

    python scripts/models/adyanth_gnb.py

Assumes features are conditionally independent and Gaussian given the class. Our features are
strongly dependent (dist_m, dist_rank and dist_minus_min all measure closeness), so this model
shows what the independence assumption costs. Scaling does not change GNB but is kept in the
Pipeline for a uniform interface. Pre-harness prototype: see _adyanth_proto.py.
"""
import sys
from pathlib import Path

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from _adyanth_proto import evaluate  # noqa: E402

CONFIG = {"var_smoothing": 1e-9, "priors": "empirical", "scaler": "standard"}

if __name__ == "__main__":
    est = Pipeline([("scale", StandardScaler()), ("clf", GaussianNB(var_smoothing=CONFIG["var_smoothing"]))])
    evaluate("adyanth_gnb", "probabilistic (generative)", est, CONFIG)
