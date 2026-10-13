"""One TGT-M training job (parallel-friendly, 1 thread): python tgtm_job.py <A|B> <cfg_idx> <seed>
Writes results/chrono/jobs/TGT-M_<v>_c<cfg>_s<seed>.json and the prediction vector (val+test rows)."""
import torch  # noqa: F401
import json
import sys
import time
from pathlib import Path

import numpy as np

import data as D
import lib as Lb

variant, ci_, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
kind = "TGT-M"
out = Path(__file__).resolve().parents[1] / "results" / "chrono" / "jobs"
out.mkdir(parents=True, exist_ok=True)
b = D.build()
m = b["meta"]
tr = (m.split_chrono == "train").to_numpy()
va = (m.split_chrono == "val").to_numpy()
te = (m.split_chrono == "test").to_numpy()
y = m["anomaly_z_capped"].to_numpy()
iva, ite = np.flatnonzero(va), np.flatnonzero(te)
B = Lb.Bundle(b, variant == "B", tr)
cfg = Lb.GRIDS[kind][ci_]
t0 = time.time()
net, _, ep = Lb.fit_deep(kind, B, y, tr, va, cfg, seed, epochs=20, patience=3)
pv, pt = Lb.predict_deep(net, B, iva), Lb.predict_deep(net, B, ite)
q = pt["q"]
rec = {"variant": variant, "cfg_idx": ci_, "seed": seed, "cfg": cfg, "epochs": ep, "time_s": time.time() - t0,
       "val_anom_MAE": float(np.abs(np.clip(pv["z"], -5, 5) - y[va]).mean()),
       "val": Lb.evaluate(m[va], pv["z"]), "test": Lb.evaluate(m[te], pt["z"]),
       "PI80_coverage": float(((y[te] >= q[:, 0]) & (y[te] <= q[:, 2])).mean()), "PI80_width": float((q[:, 2] - q[:, 0]).mean()),
       "vsn_rain_weight": float(pt["vsn_w"][:, B.is_rain].sum(1).mean()) if variant == "B" else 0.0,
       "vsn_w_mean": pt["vsn_w"].mean(0).tolist(), "attn_mean_by_step": pt["attn"].mean(0).tolist()}
full = np.full(len(m), np.nan, np.float32)
full[iva], full[ite] = pv["z"], pt["z"]
np.save(out / f"pred_TGT-M_{variant}_c{ci_}_s{seed}.npy", full)
(out / f"TGT-M_{variant}_c{ci_}_s{seed}.json").write_text(json.dumps(rec, indent=1, default=float))
print("done", variant, ci_, seed, rec["val_anom_MAE"], rec["test"]["depth_MAE"], f"{rec['time_s']:.0f}s epochs {ep}", flush=True)
