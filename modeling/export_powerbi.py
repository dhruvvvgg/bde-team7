"""Flat CSVs for Power BI / Tableau (results/powerbi/). Reads results/tables and the saved predictions."""
import torch  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import lib as Lb

ROOT = Path(__file__).resolve().parents[1] / "results"
TAB, OUT = ROOT / "tables", ROOT / "powerbi"
OUT.mkdir(exist_ok=True, parents=True)

summ = pd.read_csv(TAB / "chrono_summary.csv")
long = summ.melt(id_vars=["model", "variant"], var_name="metric_stat", value_name="value")
long[["metric", "stat"]] = long["metric_stat"].str.rsplit("_", n=1, expand=True)
long.drop(columns="metric_stat").to_csv(OUT / "model_metrics_long.csv", index=False)
for f in ("chrono_by_season", "chrono_by_completeness", "chrono_excluding_suspect", "bootstrap", "chrono_perclass_B",
          "geo_summary", "geo_per_state", "model_metrics_regression_depth", "model_metrics_regression_anomaly",
          "model_metrics_classification", "model_metrics_classification_per_class"):
    if (TAB / f"{f}.csv").exists():
        pd.read_csv(TAB / f"{f}.csv").to_csv(OUT / f"{f}.csv", index=False)

b = D.build()
m = b["meta"]
te = (m.split_chrono == "test").to_numpy()
mt = m[te].reset_index(drop=True)
out = mt[["well_id", "period_label", "state", "season", "gwl_m_bgl", "baseline_mean_m", "baseline_std_m",
          "anomaly_z_capped", "stress_category"]].copy()
for kind in ("XGBoost", "RF", "LSTM", "CNN1D", "TGT-M"):
    for v in ("A", "B"):
        f = ROOT / "chrono" / f"pred_{kind}_{v}_s0.npy"
        if f.exists():
            z = np.load(f)[te]
            out[f"{kind}_{v}_depth_pred"] = Lb.depth_from_z(mt, z).round(3)
            out[f"{kind}_{v}_stress_pred"] = np.array(D.CLASSES)[D.stress_label(z)]
out.to_csv(OUT / "test_predictions.csv", index=False)
print("exported", sorted(p.name for p in OUT.glob("*.csv")))
