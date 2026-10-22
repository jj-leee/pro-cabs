"""Build the de-identified runtime artifacts for the public PRO-CABS app.

Run locally against the private project's `app/` folder (written by
`deployment_prep.py`). It reads patient-level files that must never be committed
and writes only code-free model artifacts and aggregate summaries into this repo:

1. Copies the deployed calibrated models, the fitted preprocessing pipeline, the
   bin thresholds, the reduced feature list, `chosen_models.json` and
   `outcomes.json`.
2. `all_preds/<outcome>_<model>.parquet` (per-patient 2022-2024 test predictions +
   outcome labels) -> `population_summary/<outcome>_<model>.json`:
     * `bin_observed_rates`: observed event rate in each of the 4 risk bins;
     * `percentile_cutpoints` / `percentile_values`: a <=101-entry lookup that
       reproduces the displayed integer percentile
       (`f"{(probs < p).mean() * 100:.0f}"`) exactly, without labels or per-patient values;
     * `n_test`: test-cohort size shown in the UI.
3. SHAP explainers: tree explainers are copied (tree_path_dependent, no background
   data); a `shap.LinearExplainer` is rebuilt from the summarized background
   (mean vector + covariance) so no training rows ship. Interventional linear SHAP
   values are coef * (x - mean), so outputs are unchanged.

Usage (from the repo root, with the training-time package versions):
    python tools/build_deidentified_artifacts.py --orig "/path/to/ML-Cosmetic-Pred/app"
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

REPO = Path(__file__).resolve().parents[1]
APP = REPO / "app"


def percentile_lookup(probs: np.ndarray):
    """Exact lookup for the displayed percentile.

    The per-patient computation is ``f"{(probs < p).mean() * 100:.0f}"``. With
    s = sorted(probs), k(p) = #{s_i < p} and the displayed integer d(k) is
    monotone in k. For each displayed value d (after the first), let k_d be the
    smallest k giving d; then k(p) >= k_d  <=>  s[k_d - 1] < p. Storing the
    cut-points s[k_d - 1] (~1 per percentile point) reproduces the display exactly.
    """
    s = np.sort(np.asarray(probs, dtype=float))
    n = len(s)
    disp = []
    for k in range(n + 1):
        frac = (np.arange(n) < k).mean() * 100
        disp.append(int(f"{frac:.0f}"))
    values, cuts = [disp[0]], []
    for k in range(1, n + 1):
        if disp[k] != disp[k - 1]:
            values.append(disp[k])
            cuts.append(float(s[k - 1]))
    return values, cuts


def bin_rates(probs, labels, thresholds):
    idx = np.digitize(probs, thresholds, right=False)
    out = []
    for b in range(len(thresholds) + 1):
        m = idx == b
        out.append(None if m.sum() == 0 else float(labels[m].mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", required=True, type=Path,
                    help="private project's app/ directory")
    args = ap.parse_args()
    orig = args.orig.resolve()
    chosen = json.load(open(orig / "chosen_models.json"))

    for sub in ["models/calibrated", "preprocessors", "bin_thresholds",
                "shap_explainers", "population_summary"]:
        d = APP / sub
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    shutil.copy(orig / "chosen_models.json", APP / "chosen_models.json")
    shutil.copy(orig / "outcomes.json", APP / "outcomes.json")
    shutil.copy(orig / "preprocessors" / "pipeline.joblib", APP / "preprocessors" / "pipeline.joblib")
    shutil.copy(orig / "shap_explainers" / "feature_names.joblib",
                APP / "shap_explainers" / "feature_names.joblib")

    for outcome, model in chosen.items():
        tag = f"{outcome}_{model}"
        shutil.copy(orig / "models" / "calibrated" / f"{tag}.joblib",
                    APP / "models" / "calibrated" / f"{tag}.joblib")
        shutil.copy(orig / "bin_thresholds" / f"{tag}.npz", APP / "bin_thresholds" / f"{tag}.npz")

        df = pd.read_parquet(orig / "all_preds" / f"{tag}.parquet")
        probs = df["prob"].to_numpy()
        labels = df["label"].to_numpy()
        thr = np.load(APP / "bin_thresholds" / f"{tag}.npz")["thresholds"]
        values, cuts = percentile_lookup(probs)
        summary = {
            "outcome": outcome,
            "model": model,
            "source": "ACS-NSQIP 2022-2024 held-out test cohort (aggregate only; no patient-level rows)",
            "n_test": int(len(probs)),
            "bin_thresholds": [float(t) for t in thr],
            "bin_observed_rates": bin_rates(probs, labels, thr),
            "percentile_values": values,
            "percentile_cutpoints": cuts,
        }
        with open(APP / "population_summary" / f"{tag}.json", "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"{outcome:10s} n={len(probs)} rates={summary['bin_observed_rates']} "
              f"cutpoints={len(cuts)}")

        expl = joblib.load(orig / "shap_explainers" / f"{outcome}.joblib")
        if isinstance(expl, shap.LinearExplainer):
            bg = np.asarray(expl.masker.data, dtype=float)
            mean = np.asarray(expl.mean, dtype=float)
            cov = np.cov(bg, rowvar=False)
            new = shap.LinearExplainer(expl.model, (mean, cov),
                                       feature_names=list(expl.feature_names))
            assert np.array_equal(new.mean, expl.mean)
            assert np.isclose(new.expected_value, expl.expected_value, rtol=0, atol=1e-12)
            new.expected_value = expl.expected_value
            assert np.asarray(new.masker.data).shape[0] == 1  # only the mean row
            joblib.dump(new, APP / "shap_explainers" / f"{outcome}.joblib")
            print(f"{outcome:10s} LinearExplainer rebuilt: background {bg.shape} -> "
                  f"mean{mean.shape} + cov{cov.shape}")
        else:
            assert expl.data is None, f"{outcome} tree explainer stores background data"
            shutil.copy(orig / "shap_explainers" / f"{outcome}.joblib",
                        APP / "shap_explainers" / f"{outcome}.joblib")


if __name__ == "__main__":
    main()
