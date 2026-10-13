"""Chronological experiment for ONE model: train<=2015, tune on val (2016-17), report on test (2018-22).

usage: python run_chrono.py <model> [seeds]   model in XGBoost RF SVR ANN CNN1D LSTM TGT-M
For each variant A (no rainfall) and B (with rainfall): pick the best config on validation
(seed 0, lowest |anomaly| MAE), then refit with 3 seeds. Test rows are never used for selection.
"""
import torch  # noqa: F401
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

import data as D
import lib as Lb

kind = sys.argv[1]
ONLY = sys.argv[2] if len(sys.argv) > 2 else None  # run a single variant (A or B), written to <model>__<variant>.json
SEEDS = [int(x) for x in os.environ.get("BDE_SEEDS", "0,1,2").split(",")]   # quick mode: BDE_SEEDS=0
out = Path(os.environ.get("BDE_RESULTS", Path(__file__).resolve().parents[1] / "results")) / "chrono"   # quick mode writes elsewhere
out.mkdir(parents=True, exist_ok=True)
DEEP = kind in ("ANN", "CNN1D", "LSTM", "TGT-M")

b = D.build()
m = b["meta"]
tr = (m.split_chrono == "train").to_numpy()
va = (m.split_chrono == "val").to_numpy()
te = (m.split_chrono == "test").to_numpy()
y = m["anomaly_z_capped"].to_numpy()
iva, ite = np.flatnonzero(va), np.flatnonzero(te)
log = {"model": kind, "variants": {}}


def val_mae(zv):
    return float(np.abs(np.clip(zv, -5, 5) - y[va]).mean())


for variant, with_rain in (("A", False), ("B", True)):
    if ONLY and variant != ONLY:
        continue
    B = Lb.Bundle(b, with_rain, tr)
    rec = {"tuning": [], "seeds": []}
    t0 = time.time()
    best_i, best_v = 0, 1e9
    for i, cfg in enumerate(Lb.GRIDS[kind]):
        if DEEP:
            net, _, ep = Lb.fit_deep(kind, B, y, tr, va, cfg, 0, epochs=20, patience=3)
            v = val_mae(Lb.predict_deep(net, B, iva)["z"])
        else:
            _, [zv] = Lb.fit_predict_tabular(kind, B, y, tr, [iva], cfg, 0)
            v = val_mae(zv)
        rec["tuning"].append({"cfg": cfg, "val_anom_MAE": v})
        print(kind, variant, "cfg", i, f"val|z|MAE {v:.4f}", f"{time.time()-t0:.0f}s", flush=True)
        if v < best_v:
            best_i, best_v = i, v
    cfg = Lb.GRIDS[kind][best_i]
    rec["best_cfg"] = cfg
    zs = []
    for sd in SEEDS:
        extra = {}
        if DEEP:
            net, _, ep = Lb.fit_deep(kind, B, y, tr, va, cfg, sd, epochs=20, patience=3)
            pv, pt = Lb.predict_deep(net, B, iva), Lb.predict_deep(net, B, ite)
            zv, zt = pv["z"], pt["z"]
            if kind == "TGT-M":
                q = pt["q"]
                extra = {"PI80_coverage": float(((y[te] >= q[:, 0]) & (y[te] <= q[:, 2])).mean()),
                         "PI80_width": float((q[:, 2] - q[:, 0]).mean()),
                         "vsn_rain_weight": float(pt["vsn_w"][:, B.is_rain].sum(1).mean()) if with_rain else 0.0,
                         "vsn_w_mean": pt["vsn_w"].mean(0).tolist(),
                         "attn_mean_by_step": pt["attn"].mean(0).tolist(),
                         "epochs": ep}
        else:
            mod, [zv, zt] = Lb.fit_predict_tabular(kind, B, y, tr, [iva, ite], cfg, sd)
            if kind in ("XGBoost", "RF"):
                imp = getattr(mod, "feature_importances_", None)
                if imp is not None and sd == 0:
                    rec["importance"] = dict(zip(B.names, [float(v) for v in imp]))
        full = np.full(len(m), np.nan, np.float32)
        full[iva], full[ite] = zv, zt
        np.save(out / f"pred_{kind}_{variant}_s{sd}.npy", full)
        zs.append(zt)
        r = {"seed": sd, "val": Lb.evaluate(m[va], zv), "test": Lb.evaluate(m[te], zt), **extra}
        rec["seeds"].append(r)
        print(kind, variant, "seed", sd, "test depth MAE", round(r["test"]["depth_MAE"], 4),
              "anom R2", round(r["test"]["anom_R2"], 4), f"{time.time()-t0:.0f}s", flush=True)
    rec["time_s"] = time.time() - t0
    log["variants"][variant] = rec
    (out / (f"{kind}__{ONLY}.json" if ONLY else f"{kind}.json")).write_text(json.dumps(log, indent=1, default=float))
print("done", kind)
