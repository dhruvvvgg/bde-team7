"""Time-respecting geographic experiment for ONE model.

usage: python run_geo.py <model> <loso|g5> [A|B <fold>]   (single fold job, writes a part file)
       python run_geo.py <model> <loso|g5> merge           (combine part files, or run everything if none)
  loso: leave-one-state-out, 19 folds. g5: grouped 5-fold over states (state_fold % 5).
Each fold: train on the training period (<=2015) of the OTHER states only, early-stop/select on
the validation period (2016-17) of those same states, and test on the held-out states' test
period (2018-22). The hyperparameters are those chosen on the chronological validation split
(results/chrono), so held-out states never influence any choice.
"""
import torch  # noqa: F401
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import lib as Lb

kind, scheme = sys.argv[1], sys.argv[2]
mode = sys.argv[3] if len(sys.argv) > 3 else "all"
DEEP = kind in ("ANN", "CNN1D", "LSTM", "TGT-M")
root = Path(__file__).resolve().parents[1] / "results"
out = root / "geo"
out.mkdir(parents=True, exist_ok=True)
chrono = Lb.load_chrono(root, kind)

b = D.build()
m = b["meta"]
fold = m["state_fold"].to_numpy()
group = fold if scheme == "loso" else fold % 5
y = m["anomaly_z_capped"].to_numpy()
is_tr = (m.split_chrono == "train").to_numpy()
is_va = (m.split_chrono == "val").to_numpy()
is_te = (m.split_chrono == "test").to_numpy()
VARS = (("A", False), ("B", True))


def run_fold(variant, with_rain, g):
    cfg = chrono["variants"][variant]["best_cfg"]
    held = group == g
    tr, va, te = is_tr & ~held, is_va & ~held, is_te & held
    B = Lb.Bundle(b, with_rain, tr)
    ite = np.flatnonzero(te)
    if DEEP:
        net, _, _ = Lb.fit_deep(kind, B, y, tr, va, cfg, 0, epochs=20, patience=3)
        z = Lb.predict_deep(net, B, ite)["z"]
    else:
        _, [z] = Lb.fit_predict_tabular(kind, B, y, tr, [ite], cfg, 0)
    return ite, z


if mode in ("A", "B"):          # single-fold job
    g = int(sys.argv[4])
    t0 = time.time()
    ite, z = run_fold(mode, mode == "B", g)
    full = np.full(len(m), np.nan, np.float32)
    full[ite] = z
    np.save(out / f"part_{kind}_{scheme}_{mode}_{g}.npy", full)
    print("done", kind, scheme, mode, "fold", g, "n", len(ite), f"{time.time()-t0:.0f}s", flush=True)
    sys.exit(0)

res = {"model": kind, "scheme": scheme, "variants": {}}
for variant, with_rain in VARS:
    zhat = np.full(len(m), np.nan, np.float32)
    t0 = time.time()
    for g in sorted(set(group)):
        part = out / f"part_{kind}_{scheme}_{variant}_{g}.npy"
        if part.exists():
            p = np.load(part)
            zhat[~np.isnan(p)] = p[~np.isnan(p)]
        else:
            ite, z = run_fold(variant, with_rain, g)
            zhat[ite] = z
        print(kind, scheme, variant, "fold", g, f"{time.time()-t0:.0f}s", flush=True)
    np.save(out / f"pred_{kind}_{scheme}_{variant}.npy", zhat)
    mt = m[is_te].reset_index(drop=True)
    zt = zhat[is_te]
    assert not np.isnan(zt).any(), "some held-out rows were not predicted"
    per_state = []
    for s, idx in mt.groupby("state").groups.items():
        e = Lb.evaluate(mt.loc[idx], zt[idx])
        per_state.append({"state": s, **{k: e[k] for k in ("n", "depth_MAE", "anom_MAE", "anom_R2", "macro_F1")}})
    res["variants"][variant] = {"overall": Lb.evaluate(mt, zt), "per_state": per_state, "time_s": time.time() - t0}
    (out / f"{kind}_{scheme}.json").write_text(json.dumps(res, indent=1, default=float))
print("done", kind, scheme)
