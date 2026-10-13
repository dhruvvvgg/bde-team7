"""H1: interval quality (80% intervals). Needs results/robust/q_TGT-M_s*.npy and pred_*_p00.npy (val+test predictions)."""
import torch  # noqa: F401
import numpy as np, pandas as pd
from pathlib import Path
import xgboost as xgb
import data as D
import lib as Lb

ROOT = Path(__file__).resolve().parents[1] / "results"
b = D.build(); m = b["meta"]
tr = (m.split_chrono == "train").to_numpy(); va = (m.split_chrono == "val").to_numpy(); te = (m.split_chrono == "test").to_numpy()
y_all = m["anomaly_z_capped"].to_numpy()
yv, yt = y_all[va], y_all[te]
mt = m[te].reset_index(drop=True)
wells, _ = pd.factorize(mt["well_id"]); k = wells.max() + 1
cnt = np.bincount(wells, minlength=k).astype(float)
ALPHA = 0.2


def iscore(lo, up, y):
    return (up - lo) + (2 / ALPHA) * np.maximum(lo - y, 0) + (2 / ALPHA) * np.maximum(y - up, 0)


def cqr(lo_v, up_v, lo_t, up_t):
    e = np.maximum(lo_v - yv, yv - up_v)
    n = len(e)
    q = np.quantile(e, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n))
    return lo_t - q, up_t + q


def conformal_abs(pv, pt):
    r = np.abs(yv - np.clip(pv, -5, 5)); n = len(r)
    q = np.quantile(r, min(1.0, np.ceil((n + 1) * (1 - ALPHA)) / n))
    c = np.clip(pt, -5, 5)
    return c - q, c + q


res = {}   # name -> list of (lo, up) per seed
for s in (0, 1, 2):
    q = np.load(ROOT / "robust" / f"q_TGT-M_s{s}.npy")
    lv, uv = np.sort(q[va][:, [0, 2]], 1).T; lt, ut = np.sort(q[te][:, [0, 2]], 1).T
    res.setdefault("TGT-M native", []).append((lt, ut))
    res.setdefault("TGT-M + CQR", []).append(cqr(lv, uv, lt, ut))
    # XGBoost quantile regression (heteroscedastic baseline), best validation config, same inputs (Model B)
    cfg = Lb.load_chrono(ROOT, "XGBoost")["variants"]["B"]["best_cfg"]
    B = Lb.Bundle(b, True, tr)
    mod = xgb.XGBRegressor(tree_method="hist", n_jobs=8, random_state=s, objective="reg:quantileerror",
                           quantile_alpha=np.array([0.1, 0.9]), **cfg).fit(B.Xraw[tr], y_all[tr])
    pv, pt = mod.predict(B.Xraw[va]), mod.predict(B.Xraw[te])
    lv, uv = np.sort(pv, 1).T; lt, ut = np.sort(pt, 1).T
    res.setdefault("XGBoost-QR native", []).append((lt, ut))
    res.setdefault("XGBoost-QR + CQR", []).append(cqr(lv, uv, lt, ut))
    for name in ["SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]:
        z = np.load(ROOT / "robust" / f"pred_{name}_s{s}_p00.npy")
        res.setdefault(f"{name} point + conformal", []).append(conformal_abs(z[va], z[te]))
rows = []
for name, lst in res.items():
    cov = [((yt >= l) & (yt <= u)).mean() for l, u in lst]
    wid = [(u - l).mean() for l, u in lst]
    isc = [iscore(l, u, yt).mean() for l, u in lst]
    rows.append({"method": name, "coverage": np.mean(cov), "coverage_sd": np.std(cov), "width": np.mean(wid), "interval_score": np.mean(isc), "interval_score_sd": np.std(isc)})
tab = pd.DataFrame(rows)
rng = np.random.default_rng(0)
idx = [rng.integers(0, k, k) for _ in range(2000)]
sc = {n: np.mean([iscore(l, u, yt) for l, u in lst], 0) for n, lst in res.items()}
S = {n: np.bincount(wells, v, k) for n, v in sc.items()}
cmp_ = []
for a, c in (("TGT-M native", "XGBoost-QR native"), ("TGT-M + CQR", "XGBoost-QR + CQR")):
    d = S[a] - S[c]; bs = [d[i].sum() / cnt[i].sum() for i in idx]
    cmp_.append({"comparison": f"{a} minus {c}", "diff_interval_score": sc[a].mean() - sc[c].mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
for n in res:
    if n.endswith("point + conformal") and not n.startswith("TGT-M"):
        d = S["TGT-M native"] - S[n]; bs = [d[i].sum() / cnt[i].sum() for i in idx]
        cmp_.append({"comparison": f"TGT-M native minus {n}", "diff_interval_score": sc["TGT-M native"].mean() - sc[n].mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
tab.to_csv(ROOT / "tables" / "interval_summary.csv", index=False)
pd.DataFrame(cmp_).to_csv(ROOT / "tables" / "interval_comparison.csv", index=False)
print(tab.round(4).to_string()); print(pd.DataFrame(cmp_).round(4).to_string())
