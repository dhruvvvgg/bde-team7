"""Retrain one model (Model B, best validation config, given seed) and predict the test rows under history masking.
usage: python robust_job.py <model> <seed>   -> results/robust/pred_<model>_s<seed>_p<pp>.npy (+ q_TGT-M_s<seed>.npy, val preds at p=0)"""
import torch  # noqa: F401
import sys
import time
from pathlib import Path

import numpy as np

import data as D
import lib as Lb

kind, seed = sys.argv[1], int(sys.argv[2])
root = Path(__file__).resolve().parents[1] / "results"
out = root / "robust"
out.mkdir(parents=True, exist_ok=True)
DEEP = kind in ("ANN", "CNN1D", "LSTM", "TGT-M")
cfg = Lb.load_chrono(root, kind)["variants"]["B"]["best_cfg"]
b = D.build()
m = b["meta"]
tr = (m.split_chrono == "train").to_numpy()
va = (m.split_chrono == "val").to_numpy()
te = (m.split_chrono == "test").to_numpy()
y = m["anomaly_z_capped"].to_numpy()
iva, ite = np.flatnonzero(va), np.flatnonzero(te)
B = Lb.Bundle(b, True, tr)
t0 = time.time()
if DEEP:
    model, _, ep = Lb.fit_deep(kind, B, y, tr, va, cfg, seed, epochs=20, patience=3)
else:
    model, _ = Lb.fit_predict_tabular(kind, B, y, tr, [iva], cfg, seed)
print(kind, seed, "trained", f"{time.time()-t0:.0f}s", flush=True)
for p in (0.0, 0.2, 0.4, 0.6):
    if p == 0:
        mb = B
        drop_rate = 0.0
    else:
        Xm, sm, drop_rate = D.mask_history(b, p, 12345 + int(round(p * 100)))
        mb = Lb.with_data(B, Xm, sm)
    idx = np.concatenate([iva, ite]) if p == 0 else ite
    full = np.full(len(m), np.nan, np.float32)
    if DEEP:
        r = Lb.predict_deep(model, mb, idx)
        full[idx] = r["z"]
        if kind == "TGT-M" and p == 0:
            q = np.full((len(m), 3), np.nan, np.float32)
            q[idx] = r["q"]
            np.save(out / f"q_TGT-M_s{seed}.npy", q)
    else:
        full[idx] = Lb.predict_tabular(kind, model, mb, idx)
    np.save(out / f"pred_{kind}_s{seed}_p{int(round(p*100)):02d}.npy", full)
    print(kind, seed, "p", p, "drop share", round(drop_rate, 3), "test anom MAE", round(float(np.abs(np.clip(full[ite], -5, 5) - y[ite]).mean()), 4), flush=True)
print("done", kind, seed, flush=True)
