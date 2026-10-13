"""H2: degradation under missing recent history. Reads results/robust/pred_*.npy -> results/tables/robust_*.csv"""
import torch  # noqa: F401
import numpy as np, pandas as pd
from pathlib import Path
import data as D

ROOT = Path(__file__).resolve().parents[1] / "results"
b = D.build(); m = b["meta"]
te = (m.split_chrono == "test").to_numpy()
mt = m[te].reset_index(drop=True)
y = mt["anomaly_z_capped"].to_numpy()
wells, _ = pd.factorize(mt["well_id"]); k = wells.max() + 1
cnt = np.bincount(wells, minlength=k).astype(float)
MODELS = ["SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]
PS = (0, 20, 40, 60)
err = {}   # (model, p) -> seed-mean abs error per row
rows = []
for mod in MODELS:
    for p in PS:
        es = []
        for s in (0, 1, 2):
            f = ROOT / "robust" / f"pred_{mod}_s{s}_p{p:02d}.npy"
            if not f.exists():
                continue
            z = np.load(f)[te]
            e = np.abs(np.clip(z, -5, 5) - y); es.append(e)
            rows.append({"model": mod, "p": p / 100, "seed": s, "anom_MAE": e.mean()})
        if es:
            err[(mod, p)] = np.mean(es, 0)
df = pd.DataFrame(rows)
summ = df.groupby(["model", "p"]).anom_MAE.agg(["mean", "std", "count"]).reset_index()
summ.to_csv(ROOT / "tables" / "robust_summary.csv", index=False)
rng = np.random.default_rng(0)
idx = [rng.integers(0, k, k) for _ in range(2000)]
Dg = {mod: err[(mod, 60)] - err[(mod, 0)] for mod in MODELS if (mod, 60) in err and (mod, 0) in err}
S = {mod: np.bincount(wells, v, k) for mod, v in Dg.items()}
S60 = {mod: np.bincount(wells, err[(mod, 60)], k) for mod in Dg}
out = []
for mod in Dg:
    bs = [S[mod][i].sum() / cnt[i].sum() for i in idx]
    out.append({"comparison": mod, "what": "degradation D (p=0.6 minus p=0)", "estimate": Dg[mod].mean(), "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
for mod in Dg:
    if mod == "TGT-M" or "TGT-M" not in Dg:
        continue
    d = S["TGT-M"] - S[mod]; bs = [d[i].sum() / cnt[i].sum() for i in idx]
    out.append({"comparison": f"TGT-M minus {mod}", "what": "difference of degradations", "estimate": Dg["TGT-M"].mean() - Dg[mod].mean(),
                "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
    d = S60["TGT-M"] - S60[mod]; bs = [d[i].sum() / cnt[i].sum() for i in idx]
    out.append({"comparison": f"TGT-M minus {mod}", "what": "difference of MAE at p=0.6", "estimate": err[("TGT-M", 60)].mean() - err[(mod, 60)].mean(),
                "ci_lo": np.percentile(bs, 2.5), "ci_hi": np.percentile(bs, 97.5)})
res = pd.DataFrame(out); res.to_csv(ROOT / "tables" / "robust_degradation.csv", index=False)
print(summ.pivot(index="model", columns="p", values="mean").round(4).to_string())
print(res.round(4).to_string())
