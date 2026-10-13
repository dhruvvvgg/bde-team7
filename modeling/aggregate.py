"""Collect chronological/geographic results into tidy tables (results/tables/*.csv) + results/summary.json.
Every number placed in the report, paper and slides is read from these files by script.
"""
import torch  # noqa: F401
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data as D
import lib as Lb

ROOT = Path(__file__).resolve().parents[1] / "results"
TAB = ROOT / "tables"
TAB.mkdir(exist_ok=True, parents=True)
ORDER = ["Prev. round", "Same season LY", "Seasonal baseline", "SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]
b = D.build()
m = b["meta"]
te = (m.split_chrono == "test").to_numpy()
va = (m.split_chrono == "val").to_numpy()
mt = m[te].reset_index(drop=True)
sd = m["sd_floor"].to_numpy()
bm = m["baseline_mean_m"].to_numpy()


def naive_z():
    """Naive baselines expressed as predicted z (same transform as the models)."""
    z = {}
    z["Seasonal baseline"] = np.zeros(len(m))
    p = m["gw_prev_round_m"].fillna(m["gw_last_obs_m"]).to_numpy()
    z["Prev. round"] = np.where(np.isnan(p), 0.0, (p - bm) / sd)
    l = m["gw_same_season_lag1y_m"].to_numpy()
    z["Same season LY"] = np.where(np.isnan(l), 0.0, (l - bm) / sd)
    return z, {"prev_fallback_baseline_pct": float(np.isnan(p)[te].mean() * 100),
               "prev_fallback_lastobs_pct": float(m["gw_prev_round_m"].isna()[te].mean() * 100),
               "lag1y_fallback_pct": float(np.isnan(l)[te].mean() * 100)}


NZ, NZ_INFO = naive_z()
rows, cls_rows, cm, seed_rows = [], [], {}, []
preds = {}  # (model, variant) -> list of z arrays over test rows (one per seed)
predsv = {}  # same for validation rows (used only to calibrate class thresholds)
for name, z in NZ.items():
    for v in ("A", "B"):
        e = Lb.evaluate(mt, z[te])
        rows.append({"model": name, "variant": v, "seed": 0, **e})
        preds[(name, v)] = [z[te]]
        predsv[(name, v)] = [z[va]]
for name in ORDER:
    j = Lb.load_chrono(ROOT, name)
    if not j["variants"]:
        continue
    for v, rec in j["variants"].items():
        arrs, arrsv = [], []
        for s in rec["seeds"]:
            full = np.load(ROOT / "chrono" / f"pred_{name}_{v}_s{s['seed']}.npy")
            arrs.append(full[te])
            arrsv.append(full[va])
            rows.append({"model": name, "variant": v, "seed": s["seed"], **s["test"]})
            seed_rows.append({"model": name, "variant": v, "seed": s["seed"],
                              **{k: val for k, val in s.items() if k in ("PI80_coverage", "PI80_width", "vsn_rain_weight")}})
        preds[(name, v)] = arrs
        predsv[(name, v)] = arrsv
df = pd.DataFrame(rows)
df.to_csv(TAB / "chrono_per_seed.csv", index=False)
metrics = [c for c in df.columns if c not in ("model", "variant", "seed")]
agg = df.groupby(["model", "variant"])[metrics].agg(["mean", "std"])
agg.columns = [f"{a}_{b_}" for a, b_ in agg.columns]
agg = agg.reset_index()
agg["_o"] = agg["model"].map({k: i for i, k in enumerate(ORDER)})
agg = agg.sort_values(["_o", "variant"]).drop(columns="_o")
agg.to_csv(TAB / "chrono_summary.csv", index=False)




# ---------- class thresholds calibrated on VALIDATION only (macro-F1), then applied to test
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
mv = m[va].reset_index(drop=True)
ytv, ytt = D.true_label(mv["stress_category"]), D.true_label(mt["stress_category"])
GRID = [(a, b2, c) for a in np.arange(0.1, 1.31, 0.1) for b2 in np.arange(0.3, 1.91, 0.1) for c in np.arange(0.5, 2.51, 0.1)
        if a < b2 < c]


def lab(z, th):
    return np.select([z >= th[2], z >= th[1], z >= th[0]], [3, 2, 1], default=0)


def fast_f1(y, yp):
    cm = np.bincount(y * 4 + yp, minlength=16).reshape(4, 4).astype(float)
    tp = np.diag(cm)
    den = cm.sum(0) + cm.sum(1)
    return float(np.mean(np.where(den > 0, 2 * tp / np.maximum(den, 1), 0)))


def calibrate(zv):
    best, bt = -1, None
    for th in GRID:
        f = fast_f1(ytv, lab(zv, th))
        if f > best:
            best, bt = f, th
    return bt


cal_rows, cal_cm = [], {}
for (name, v), arrs in preds.items():
    for si, zt in enumerate(arrs):
        th = calibrate(predsv[(name, v)][si])
        yp = lab(zt, th)
        p, r, f, _ = precision_recall_fscore_support(ytt, yp, labels=[0, 1, 2, 3], zero_division=0)
        row = {"model": name, "variant": v, "seed": si, "t1": th[0], "t2": th[1], "t3": th[2],
               "cal_acc": accuracy_score(ytt, yp),
               "cal_macro_F1": f1_score(ytt, yp, average="macro", labels=[0, 1, 2, 3], zero_division=0)}
        for i, c in enumerate(D.CLASSES):
            row[f"cal_P_{c}"], row[f"cal_R_{c}"], row[f"cal_F1_{c}"] = p[i], r[i], f[i]
        cal_rows.append(row)
        if si == 0:
            cal_cm[f"{name}|{v}"] = confusion_matrix(ytt, yp, labels=[0, 1, 2, 3]).tolist()
calc = pd.DataFrame(cal_rows)
calc.to_csv(TAB / "chrono_calibrated_per_seed.csv", index=False)
calagg = calc.drop(columns="seed").groupby(["model", "variant"]).agg(["mean", "std"])
calagg.columns = [f"{a}_{b_}" for a, b_ in calagg.columns]
calagg = calagg.reset_index()
calagg["_o"] = calagg["model"].map({k: i for i, k in enumerate(ORDER)})
calagg.sort_values(["_o", "variant"]).drop(columns="_o").to_csv(TAB / "chrono_calibrated.csv", index=False)
(TAB / "confusion_matrices_calibrated.json").write_text(json.dumps(cal_cm))


# ---------- confusion matrices (seed 0), per season, per-class
def seed0(name, v):
    return preds[(name, v)][0]


for (name, v), arrs in preds.items():
    cm[f"{name}|{v}"] = Lb.confusion(mt, arrs[0]).tolist()
(TAB / "confusion_matrices.json").write_text(json.dumps(cm))

season_rows = []
for (name, v), arrs in preds.items():
    for sea, idx in mt.groupby("season", observed=True).groups.items():
        maes = [Lb.evaluate(mt.loc[idx], z[idx])["depth_MAE"] for z in arrs]
        am = [Lb.evaluate(mt.loc[idx], z[idx])["anom_MAE"] for z in arrs]
        f1 = [Lb.evaluate(mt.loc[idx], z[idx])["macro_F1"] for z in arrs]
        season_rows.append({"model": name, "variant": v, "season": sea, "n": len(idx),
                            "depth_MAE": np.mean(maes), "anom_MAE": np.mean(am), "macro_F1": np.mean(f1)})
pd.DataFrame(season_rows).to_csv(TAB / "chrono_by_season.csv", index=False)

# per-state chronological errors (seed 0), to compare with the unseen-state results
psr = []
for (name, v), arrs in preds.items():
    if name in NZ:
        continue
    for st_, idx in mt.groupby("state").groups.items():
        e = Lb.evaluate(mt.loc[idx], arrs[0][idx])
        psr.append({"model": name, "variant": v, "state": st_, "n": len(idx), "depth_MAE": e["depth_MAE"], "anom_MAE": e["anom_MAE"]})
pd.DataFrame(psr).to_csv(TAB / "chrono_per_state.csv", index=False)

# completeness bands (share of earlier rounds with a reading)
band = pd.cut(mt["completeness_to_date"], [0, 0.7, 0.85, 0.95, 1.0], include_lowest=True,
              labels=["<70%", "70-85%", "85-95%", ">95%"])
cb = []
for (name, v), arrs in preds.items():
    for lab in band.cat.categories:
        idx = np.flatnonzero((band == lab).to_numpy())
        if len(idx) == 0:
            continue
        cb.append({"model": name, "variant": v, "band": lab, "n": len(idx),
                   "depth_MAE": np.mean([Lb.evaluate(mt.loc[idx], z[idx])["depth_MAE"] for z in arrs])})
pd.DataFrame(cb).to_csv(TAB / "chrono_by_completeness.csv", index=False)

# suspect-reading sensitivity (exclude flagged rows)
ok = ~mt["suspect_reading"].fillna(False).to_numpy(bool)
sr = []
for (name, v), arrs in preds.items():
    e = [Lb.evaluate(mt[ok].reset_index(drop=True), z[ok]) for z in arrs]
    sr.append({"model": name, "variant": v, "n": int(ok.sum()),
               "depth_MAE": np.mean([x["depth_MAE"] for x in e]), "anom_R2": np.mean([x["anom_R2"] for x in e]),
               "macro_F1": np.mean([x["macro_F1"] for x in e])})
pd.DataFrame(sr).to_csv(TAB / "chrono_excluding_suspect.csv", index=False)


# ---------- paired bootstrap over wells
def boot(abs_a, abs_b, wells, n=2000, seed=0):
    codes, uniq = pd.factorize(wells)
    k = len(uniq)
    da, db_ = np.bincount(codes, abs_a, k), np.bincount(codes, abs_b, k)
    cnt = np.bincount(codes, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    d = []
    for _ in range(n):
        s = rng.integers(0, k, k)
        d.append((db_[s].sum() - da[s].sum()) / cnt[s].sum())
    d = np.array(d)
    return float((abs_b.mean() - abs_a.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def abs_err(name, v, kind):
    out = []
    for z in preds[(name, v)]:
        if kind == "depth":
            out.append(np.abs(Lb.depth_from_z(mt, z) - mt["gwl_m_bgl"].to_numpy()))
        else:
            out.append(np.abs(np.clip(z, -5, 5) - mt["anomaly_z_capped"].to_numpy()))
    return np.mean(out, 0)


bt = []
wells = mt["well_id"].to_numpy()
for name in ORDER:
    if name in NZ:
        continue  # naive baselines have identical A and B
    if (name, "A") in preds and (name, "B") in preds:
        for kind in ("depth", "anom"):
            d, lo, hi = boot(abs_err(name, "A", kind), abs_err(name, "B", kind), wells)
            bt.append({"comparison": f"{name}: B - A (rainfall effect)", "metric": f"{kind}_MAE", "diff": d,
                       "ci_lo": lo, "ci_hi": hi})
if ("TGT-M", "B") in preds:
    for other in ORDER:
        if other in ("TGT-M",) or (other, "B") not in preds:
            continue
        for kind in ("depth", "anom"):
            d, lo, hi = boot(abs_err(other, "B", kind), abs_err("TGT-M", "B", kind), wells)
            bt.append({"comparison": f"TGT-M(B) - {other}(B)", "metric": f"{kind}_MAE", "diff": d,
                       "ci_lo": lo, "ci_hi": hi})
pd.DataFrame(bt).to_csv(TAB / "bootstrap.csv", index=False)

# ---------- per-class table for B variant
pc = agg[agg.variant == "B"][["model"] + [c for c in agg.columns if c.startswith(("P_", "R_", "F1_")) and c.endswith("_mean")]]
pc.to_csv(TAB / "chrono_perclass_B.csv", index=False)

# ---------- geographic
g = []
for f in sorted((ROOT / "geo").glob("*.json")) if (ROOT / "geo").exists() else []:
    j = json.loads(f.read_text())
    for v, rec in j["variants"].items():
        g.append({"model": j["model"], "scheme": j["scheme"], "variant": v, **rec["overall"]})
if g:
    pd.DataFrame(g).to_csv(TAB / "geo_summary.csv", index=False)
    ps = []
    for f in sorted((ROOT / "geo").glob("*.json")):
        j = json.loads(f.read_text())
        for v, rec in j["variants"].items():
            for r in rec["per_state"]:
                ps.append({"model": j["model"], "scheme": j["scheme"], "variant": v, **r})
    pd.DataFrame(ps).to_csv(TAB / "geo_per_state.csv", index=False)

# naive-baseline reference to geo not needed (no training).  TGT-M diagnostics
tg = Lb.load_chrono(ROOT, "TGT-M")
summ = {"n_test": int(te.sum()), "n_val": int(va.sum()), "n_train": int((m.split_chrono == "train").sum()),
        "naive_info": NZ_INFO, "best_cfg": {}}
for kname in ORDER:
    j = Lb.load_chrono(ROOT, kname)
    if not j["variants"]:
        continue
    summ["best_cfg"][j["model"]] = {v: r["best_cfg"] for v, r in j["variants"].items()}
    summ.setdefault("tuning", {})[j["model"]] = {v: r["tuning"] for v, r in j["variants"].items()}
if tg["variants"]:
    j = tg
    summ["tgtm"] = {v: {"PI80_coverage": float(np.mean([s["PI80_coverage"] for s in r["seeds"]])),
                        "PI80_width": float(np.mean([s["PI80_width"] for s in r["seeds"]])),
                        "vsn_rain_weight": float(np.mean([s["vsn_rain_weight"] for s in r["seeds"]])),
                        "attn_mean_by_step": np.mean([s["attn_mean_by_step"] for s in r["seeds"]], 0).tolist(),
                        "vsn_w_mean": np.mean([s["vsn_w_mean"] for s in r["seeds"]], 0).tolist()}
                    for v, r in j["variants"].items()}
(ROOT / "summary.json").write_text(json.dumps(summ, indent=1, default=float))
print(agg[["model", "variant", "depth_MAE_mean", "anom_MAE_mean", "anom_R2_mean", "macro_F1_mean", "AUC_stressed_mean"]].to_string())
