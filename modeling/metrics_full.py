"""Full model metrics on the test split (2018-2022), from the saved predictions of every model (3 seeds).

Regression (depth in metres, and the anomaly z): MSE, RMSE, MAE, MAPE, R2, NRMSE, MedAE, Bias.
Classification (4 stress classes; predicted class = thresholds 1 / 1.5 / 2 on the predicted z, and the validation-calibrated
thresholds): accuracy, precision, recall, F1, specificity, AUC-ROC, log loss, MCC, Cohen's kappa, balanced accuracy.

Writes results/tables/model_metrics_{regression_depth,regression_anomaly,classification,classification_per_class}.csv
Definitions match src/gwstress/error_metrics.py (e = prediction - actual; MAPE only where |actual| >= 1 m).
"""
import torch  # noqa: F401
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (balanced_accuracy_score, cohen_kappa_score, confusion_matrix, matthews_corrcoef, roc_auc_score)

import data as D
import lib as Lb

ROOT = Path(__file__).resolve().parents[1] / "results"
TAB = ROOT / "tables"
ORDER = ["Prev. round", "Same season LY", "Seasonal baseline", "SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]
MAPE_MIN = 1.0
CLS = D.CLASSES

b = D.build()
m = b["meta"]
te = (m.split_chrono == "test").to_numpy()
va = (m.split_chrono == "val").to_numpy()
mt, mv = m[te].reset_index(drop=True), m[va].reset_index(drop=True)
y_depth, y_z = mt["gwl_m_bgl"].to_numpy(), mt["anomaly_z_capped"].to_numpy()
yt_cls, yv_cls = D.true_label(mt["stress_category"]), D.true_label(mv["stress_category"])


def reg(y, p, mape=True):
    e = p - y
    mse = float((e ** 2).mean())
    rmse = float(np.sqrt(mse))
    rng = float(y.max() - y.min())
    big = np.abs(y) >= MAPE_MIN
    return {"n": len(y), "MSE": mse, "RMSE": rmse, "MAE": float(np.abs(e).mean()),
            "MAPE_pct": float(100 * (np.abs(e[big]) / np.abs(y[big])).mean()) if mape and big.any() else np.nan,
            "R2": float(1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum()), "NRMSE_range": rmse / rng,
            "NRMSE_mean": rmse / float(np.abs(y.mean())) if mape else np.nan, "MedAE": float(np.median(np.abs(e))), "Bias": float(e.mean())}


def lab(z, th):
    return np.select([z >= th[2], z >= th[1], z >= th[0]], [3, 2, 1], default=0)


def f1_macro(y, p):
    cm = np.bincount(y * 4 + p, minlength=16).reshape(4, 4).astype(float)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * np.diag(cm) / np.maximum(den, 1), 0)))


GRID = [(a, b2, c) for a in np.arange(0.1, 1.31, 0.1) for b2 in np.arange(0.3, 1.91, 0.1) for c in np.arange(0.5, 2.51, 0.1) if a < b2 < c]


def calibrate(zv):
    return max(GRID, key=lambda th: f1_macro(yv_cls, lab(zv, th)))


def cls_metrics(yt, yp, z):
    cm = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    tn = cm.sum() - tp - fp - fn
    prec, rec = tp / np.maximum(tp + fp, 1), tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    spec = tn / np.maximum(tn + fp, 1)
    s = (yt >= 2).astype(int)
    out = {"accuracy": float(tp.sum() / cm.sum()), "precision_macro": float(prec.mean()), "recall_macro": float(rec.mean()), "F1_macro": float(f1.mean()),
           "specificity_macro": float(spec.mean()), "AUC_ROC_stressed": float(roc_auc_score(s, z)), "AUC_ROC_High": float(roc_auc_score((yt == 3).astype(int), z)),
           "MCC": float(matthews_corrcoef(yt, yp)), "kappa": float(cohen_kappa_score(yt, yp)), "balanced_accuracy": float(balanced_accuracy_score(yt, yp))}
    per = {c: {"precision": prec[i], "recall": rec[i], "F1": f1[i], "specificity": spec[i], "support": int(cm[i].sum())} for i, c in enumerate(CLS)}
    return out, per


# ---- naive baselines as predicted z (same transform as the models)
sd, bm = m["sd_floor"].to_numpy(), m["baseline_mean_m"].to_numpy()
prev = m["gw_prev_round_m"].fillna(m["gw_last_obs_m"]).to_numpy()
lag = m["gw_same_season_lag1y_m"].to_numpy()
naive = {"Seasonal baseline": np.zeros(len(m)), "Prev. round": np.where(np.isnan(prev), 0.0, (prev - bm) / sd),
         "Same season LY": np.where(np.isnan(lag), 0.0, (lag - bm) / sd)}

reg_d, reg_z, cls_rows, per_rows = [], [], [], []
for name in ORDER:
    for v in ("A", "B"):
        series = []
        if name in naive:
            if v == "A":
                z = naive[name]
                series = [(0, z[te], z[va])]
        else:
            for s in (0, 1, 2):
                f = ROOT / "chrono" / f"pred_{name}_{v}_s{s}.npy"
                if f.exists():
                    z = np.load(f)
                    series.append((s, z[te], z[va]))
        for s, zt, zv in series:
            vv = "A/B" if name in naive else v
            reg_d.append({"model": name, "variant": vv, "seed": s, **reg(y_depth, Lb.depth_from_z(mt, zt))})
            reg_z.append({"model": name, "variant": vv, "seed": s, **reg(y_z, np.clip(zt, -5, 5), mape=False)})
            th_cal = calibrate(zv)
            for tname, th in (("default 1/1.5/2", (1.0, 1.5, 2.0)), ("validation-calibrated", th_cal)):
                out, per = cls_metrics(yt_cls, lab(zt, th), zt)
                cls_rows.append({"model": name, "variant": vv, "seed": s, "thresholds": tname, **out})
                for c, d in per.items():
                    per_rows.append({"model": name, "variant": vv, "seed": s, "thresholds": tname, "class": c, **d})


def summarise(rows, keys):
    df = pd.DataFrame(rows).drop(columns="seed")
    g = df.groupby(keys, sort=False)
    mean, std = g.mean(numeric_only=True), g.std(numeric_only=True)
    out = mean.copy()
    for c in mean.columns:
        out[c + "_sd"] = std[c]
    return out.reset_index()


summarise(reg_d, ["model", "variant"]).to_csv(TAB / "model_metrics_regression_depth.csv", index=False)
summarise(reg_z, ["model", "variant"]).to_csv(TAB / "model_metrics_regression_anomaly.csv", index=False)
summarise(cls_rows, ["model", "variant", "thresholds"]).to_csv(TAB / "model_metrics_classification.csv", index=False)
summarise(per_rows, ["model", "variant", "thresholds", "class"]).to_csv(TAB / "model_metrics_classification_per_class.csv", index=False)
d = pd.read_csv(TAB / "model_metrics_regression_depth.csv")
print(d[d.variant != "A"][["model", "variant", "RMSE", "MSE", "MAE", "MAPE_pct", "R2", "NRMSE_range", "MedAE", "Bias"]].round(3).to_string(index=False))
c = pd.read_csv(TAB / "model_metrics_classification.csv")
print(c[(c.variant != "A") & (c.thresholds == "validation-calibrated")][["model", "accuracy", "precision_macro", "recall_macro", "F1_macro", "specificity_macro", "AUC_ROC_stressed", "MCC", "kappa", "balanced_accuracy"]].round(3).to_string(index=False))
