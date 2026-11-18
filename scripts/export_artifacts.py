"""Export the selected classifier to artifacts/ for pathsnap.pipeline.predict_trip and the API.

    python scripts/export_artifacts.py                 # pick by the selection rule below
    python scripts/export_artifacts.py advitha_xgb     # or name a model explicitly (e.g. after tuning)

Selection rule: highest **cross-validated** observation-level Top-1 on the 80 train trips
(the primary metric) among the non-ablation models in experiments/results/models/. Student-test
numbers are recorded but play no part in the choice.

The model is rebuilt from ``scripts/models/<name>.py::build()`` and fitted on every reliable
training observation (the same data as the harness's final fit). Writes
  artifacts/model.joblib      the fitted estimator (predict_proba)
  artifacts/metadata.json     name, config, seed, features, selection evidence, library versions
"""
from __future__ import annotations

import importlib.util
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np

from pathsnap.config import load_config, repo_path
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.harness import SEED, load_split


def model_results() -> dict[str, dict]:
    out = {}
    for p in sorted(repo_path("experiments/results/models").glob("*.json")):
        d = json.loads(p.read_text())
        if not d["name"].startswith("ablation_") and d.get("features") == FEATURE_COLUMNS:
            out[d["name"]] = d
    return out


def select(results: dict[str, dict]) -> str:
    return max(results, key=lambda n: (results[n]["cv"]["top1_mean"], -results[n]["cv"]["log_loss_mean"]))


def build_from_script(name: str):
    path = repo_path(f"scripts/models/{name}.py")
    spec = importlib.util.spec_from_file_location(f"model_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.build(), mod.CONFIG


def versions() -> dict:
    import pandas
    import sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
         "scikit-learn": sklearn.__version__}
    for lib in ("lightgbm", "xgboost"):
        try:
            v[lib] = __import__(lib).__version__
        except ImportError:
            pass
    return v


def main(argv: list[str]) -> int:
    results = model_results()
    name = argv[0] if argv else select(results)
    if name not in results:
        raise SystemExit(f"No harness result for {name!r}; run scripts/models/{name}.py first")
    ranking = sorted(results, key=lambda n: -results[n]["cv"]["top1_mean"])
    print("CV Top-1 ranking:", ", ".join(f"{n} {results[n]['cv']['top1_mean']:.4f}" for n in ranking))
    print(f"Exporting {name} ({'explicit' if argv else 'selection rule: best CV Top-1'})")

    est, config = build_from_script(name)
    tr, info = load_split("train")
    t0 = time.perf_counter()
    est.fit(tr[FEATURE_COLUMNS].to_numpy(np.float32), tr["label"].to_numpy())
    fit_s = time.perf_counter() - t0

    out = repo_path("artifacts/model.joblib")
    out.parent.mkdir(exist_ok=True)
    joblib.dump(est, out, compress=3)
    r = results[name]
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=repo_path(".")).stdout.strip()
    except OSError:
        commit = None
    meta = {
        "model": name, "family": r["family"], "config": config, "seed": SEED,
        "features": FEATURE_COLUMNS, "feature_list": "artifacts/feature_list.json",
        "candidates": load_config("candidates"),
        "selection": {"rule": "explicit" if argv else "highest CV observation-level Top-1 (train, 5-fold GroupKFold by trip)",
                      "cv_top1_ranking": {n: results[n]["cv"]["top1_mean"] for n in ranking}},
        "cv": {k: r["cv"][k] for k in ("macro_f1_mean", "macro_f1_std", "top1_mean", "top1_std", "log_loss_mean")},
        "student_test": {k: r["student_test"][k] for k in ("top1", "macro_f1", "log_loss", "route_continuity_argmax")},
        "trained_on": {"partition": "train", **info}, "fit_seconds": fit_s,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_commit": commit,
        "versions": versions(),
    }
    repo_path("artifacts/metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB) and artifacts/metadata.json; fit {fit_s:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
