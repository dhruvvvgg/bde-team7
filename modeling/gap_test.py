import torch  # noqa: F401
import numpy as np, pandas as pd, json
from pathlib import Path
import data as D
ROOT = Path(__file__).resolve().parents[1] / "results"
b = D.build(); m = b["meta"]
te = (m.split_chrono == "test").to_numpy()
mt = m[te].reset_index(drop=True)
y = mt["anomaly_z_capped"].to_numpy()
wells, _ = pd.factorize(mt["well_id"]); k = wells.max() + 1
cnt = np.bincount(wells, minlength=k).astype(float)
def err(z): return np.abs(np.clip(z, -5, 5) - y)
d = {}
for mod in ["SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]:
    zs = np.load(ROOT / "chrono" / f"pred_{mod}_B_s0.npy")[te]
    zg = np.load(ROOT / "geo" / f"pred_{mod}_g5_B.npy")[te]
    d[mod] = err(zg) - err(zs)          # per-row increase in error when the state is unseen
rng = np.random.default_rng(0)
idx = [rng.integers(0, k, k) for _ in range(2000)]
S = {mod: np.bincount(wells, v, k) for mod, v in d.items()}
out = []
for mod in d:
    est = d[mod].mean(); bs = [S[mod][i].sum() / cnt[i].sum() for i in idx]
    out.append((mod, "own gap", est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
for mod in d:
    if mod == "TGT-M": continue
    diff = S["TGT-M"] - S[mod]; est = d["TGT-M"].mean() - d[mod].mean()
    bs = [diff[i].sum() / cnt[i].sum() for i in idx]
    out.append((f"TGT-M minus {mod}", "gap diff", est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
df = pd.DataFrame(out, columns=["comparison", "what", "estimate", "ci_lo", "ci_hi"])
df.to_csv(ROOT / "tables" / "generalization_gap.csv", index=False)
print(df.round(4).to_string())
