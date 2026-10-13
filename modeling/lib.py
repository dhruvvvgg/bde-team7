"""Training, prediction and metric helpers shared by every experiment."""
from __future__ import annotations

import torch  # noqa: F401  (must load before numpy/pandas on Windows)
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
                             roc_auc_score)

import data as D
import nets as N

def _no_power_throttling():
    """Windows 11 puts windowless background processes into efficiency mode (about 3x slower); opt this process out."""
    try:
        import ctypes
        from ctypes import wintypes

        class _S(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]
        k = ctypes.WinDLL("kernel32")
        k.GetCurrentProcess.restype = wintypes.HANDLE
        k.SetProcessInformation(k.GetCurrentProcess(), 4, ctypes.byref(_S(1, 1, 0)), ctypes.sizeof(_S))
    except Exception:
        pass


_no_power_throttling()
NT = int(os.environ.get("BDE_THREADS", 8))
torch.set_num_threads(NT)


# ------------------------------------------------------------------ metrics
def depth_from_z(meta: pd.DataFrame, z: np.ndarray) -> np.ndarray:
    return np.maximum(meta["baseline_mean_m"].to_numpy() + z * meta["sd_floor"].to_numpy(), 0.0)


def _reg(y, p):
    e = p - y
    ss = ((y - y.mean()) ** 2).sum()
    return {"MAE": float(np.abs(e).mean()), "RMSE": float(np.sqrt((e ** 2).mean())),
            "R2": float(1 - (e ** 2).sum() / ss) if ss > 0 else np.nan}


def evaluate(meta: pd.DataFrame, z_hat: np.ndarray) -> dict:
    """All headline metrics for one set of rows (meta rows aligned with z_hat)."""
    z_hat = np.asarray(z_hat, float)
    out = {"n": int(len(meta))}
    d = _reg(meta["gwl_m_bgl"].to_numpy(), depth_from_z(meta, z_hat))
    out.update({f"depth_{k}": v for k, v in d.items()})
    a = _reg(meta["anomaly_z_capped"].to_numpy(), np.clip(z_hat, -5, 5))
    out.update({f"anom_{k}": v for k, v in a.items()})
    yt = D.true_label(meta["stress_category"])
    yp = D.stress_label(z_hat)
    out["acc"] = float(accuracy_score(yt, yp))
    out["macro_F1"] = float(f1_score(yt, yp, average="macro", labels=[0, 1, 2, 3], zero_division=0))
    p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=[0, 1, 2, 3], zero_division=0)
    for i, c in enumerate(D.CLASSES):
        out[f"P_{c}"], out[f"R_{c}"], out[f"F1_{c}"] = float(p[i]), float(r[i]), float(f[i])
    s = (yt >= 2).astype(int)
    out["AUC_stressed"] = float(roc_auc_score(s, z_hat)) if 0 < s.sum() < len(s) else np.nan
    return out


def confusion(meta, z_hat) -> np.ndarray:
    return confusion_matrix(D.true_label(meta["stress_category"]), D.stress_label(z_hat), labels=[0, 1, 2, 3])


# ------------------------------------------------------------------ tensors
class Bundle:
    """Train-fitted scaling for one feature set (A = no rainfall, B = with rainfall)."""

    def __init__(self, b: dict, with_rain: bool, fit_mask: np.ndarray):
        self.with_rain = with_rain
        keep = np.ones(len(b["names"]), bool) if with_rain else ~b["is_rain"]
        self.keep = keep
        self.names = [n for n, k in zip(b["names"], keep) if k]
        X = b["X"].to_numpy(float)[:, keep]
        self.sc = D.Scaler().fit(X[fit_mask])
        self.Xraw = X
        self.X = self.sc.transform(X)
        self.is_rain = np.concatenate([b["is_rain"][keep], np.zeros(int(self.sc.flag.sum()), bool)])
        seq = b["seq"].copy()
        if with_rain:
            tr = seq[fit_mask][:, :, 7:9]
            mu, sd = tr[seq[fit_mask][:, :, 6] > 0].mean(0), tr[seq[fit_mask][:, :, 6] > 0].std(0)
            self.rmu, self.rsd = mu, sd
            seq[:, :, 7:9] = np.where(seq[:, :, 6:7] > 0, (seq[:, :, 7:9] - mu) / sd, 0)
        else:
            seq[:, :, 7:9] = 0
        self.seq = seq.astype(np.float32)


def with_data(bundle, X_df, seq_raw):
    """Same train-fitted scaling applied to (masked) raw arrays; used by the missing-history stress test."""
    import copy
    nb = copy.copy(bundle)
    X = X_df.to_numpy(float)[:, bundle.keep]
    nb.Xraw, nb.X = X, bundle.sc.transform(X)
    seq = seq_raw.copy()
    if bundle.with_rain:
        seq[:, :, 7:9] = np.where(seq[:, :, 6:7] > 0, (seq[:, :, 7:9] - bundle.rmu) / bundle.rsd, 0)
    else:
        seq[:, :, 7:9] = 0
    nb.seq = seq.astype(np.float32)
    return nb


# ------------------------------------------------------------------ deep training
def fit_deep(kind, bundle, y, tr, va, cfg, seed, epochs=30, patience=4, verbose=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rain_idx = np.flatnonzero(bundle.is_rain)
    net = N.NETS[kind](n_static=bundle.X.shape[1], rain_idx=rain_idx, **cfg["arch"])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 1e-4))
    Xt, St = torch.from_numpy(bundle.X), torch.from_numpy(bundle.seq)
    yt = torch.from_numpy(y.astype(np.float32))
    huber = nn.HuberLoss(delta=1.0)
    tri, vai = np.flatnonzero(tr), np.flatnonzero(va)
    best, best_state, bad = 1e9, None, 0
    bs = 1024
    for ep in range(epochs):
        net.train()
        perm = np.random.permutation(tri)
        for i in range(0, len(perm), bs):
            j = torch.from_numpy(perm[i:i + bs])
            if kind.startswith("TGT-M"):
                pt, q = net(St[j], Xt[j], return_all=True)
                loss = huber(pt, yt[j]) + 0.5 * N.quantile_loss(q, yt[j])
            else:
                loss = huber(net(St[j], Xt[j]), yt[j])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        net.eval()
        with torch.no_grad():
            pv = torch.cat([net(St[vai[i:i + 4096]], Xt[vai[i:i + 4096]]) for i in range(0, len(vai), 4096)])
            vl = float((pv - yt[vai]).abs().mean())
        if verbose:
            print(f"  ep{ep} val |z| MAE {vl:.4f}", flush=True)
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    return net, best, ep + 1


def predict_deep(net, bundle, idx, kind=None):
    Xt, St = torch.from_numpy(bundle.X), torch.from_numpy(bundle.seq)
    out, q, w, a = [], [], [], []
    with torch.no_grad():
        for i in range(0, len(idx), 4096):
            j = torch.from_numpy(idx[i:i + 4096])
            if isinstance(net, N.TGTM):
                pt, qq = net(St[j], Xt[j], return_all=True)
                out.append(pt); q.append(qq); w.append(net.last_w)
                a.append(net.last_attn.squeeze(1))
            else:
                out.append(net(St[j], Xt[j]))
    res = {"z": torch.cat(out).numpy()}
    if q:
        res["q"] = torch.cat(q).numpy()
        res["vsn_w"] = torch.cat(w).numpy()
        res["attn"] = torch.cat(a).numpy()
    return res


# ------------------------------------------------------------------ tabular models
def make_tabular(kind, cfg, seed):
    if kind == "XGBoost":
        import xgboost as xgb
        return xgb.XGBRegressor(tree_method="hist", n_jobs=NT, random_state=seed, objective="reg:absoluteerror", **cfg)
    if kind == "RF":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_jobs=NT, random_state=seed, **cfg)
    if kind == "SVR":
        from sklearn.kernel_approximation import Nystroem
        from sklearn.pipeline import make_pipeline
        from sklearn.svm import LinearSVR
        return make_pipeline(Nystroem(n_components=cfg["nys"], gamma=cfg["gamma"], random_state=seed),
                             LinearSVR(C=cfg["C"], epsilon=0.1, loss="epsilon_insensitive", dual="auto",
                                       max_iter=3000, random_state=seed))
    raise KeyError(kind)


def fit_predict_tabular(kind, bundle, y, tr, idx_list, cfg, seed):
    """XGBoost / RF use raw (NaN-aware where supported); SVR uses the scaled imputed matrix."""
    if kind == "SVR":
        M = bundle.X
    elif kind == "RF":
        M = np.where(np.isnan(bundle.Xraw), -999.0, bundle.Xraw)   # sentinel: RF splits on 'missing'
    else:
        M = bundle.Xraw
    m = make_tabular(kind, cfg, seed)
    m.fit(M[tr], y[tr])
    return m, [m.predict(M[i]) for i in idx_list]


def predict_tabular(kind, model, bundle, idx):
    if kind == "SVR":
        M = bundle.X
    elif kind == "RF":
        M = np.where(np.isnan(bundle.Xraw), -999.0, bundle.Xraw)
    else:
        M = bundle.Xraw
    return model.predict(M[idx])


GRIDS = {
    "XGBoost": [dict(n_estimators=n, max_depth=d, learning_rate=lr, subsample=0.8, colsample_bytree=0.8,
                     min_child_weight=mcw, reg_lambda=lam)
                for n, d, lr, mcw, lam in [(300, 3, 0.05, 20, 5), (300, 4, 0.05, 5, 1), (500, 4, 0.03, 50, 10),
                                           (400, 5, 0.03, 100, 10)]],
    "RF": [dict(n_estimators=150, max_depth=d, min_samples_leaf=m, max_features=f)
           for d, m, f in [(12, 20, 0.5), (16, 10, 0.33), (None, 30, 0.33), (20, 50, 0.5)]],
    "SVR": [dict(nys=n, gamma=g, C=c) for n, g, c in [(400, 0.02, 0.5), (800, 0.01, 1.0), (800, 0.02, 0.1), (1200, 0.005, 1.0)]],
    "ANN": [dict(arch=dict(hidden=h, drop=dr), lr=lr)
            for h, dr, lr in [(64, 0.1, 2e-3), (128, 0.2, 1e-3), (64, 0.0, 3e-3)]],
    "CNN1D": [dict(arch=dict(hidden=h, drop=dr), lr=lr)
              for h, dr, lr in [(32, 0.1, 2e-3), (64, 0.1, 1e-3), (32, 0.2, 3e-3)]],
    "LSTM": [dict(arch=dict(hidden=h, drop=dr), lr=lr)
             for h, dr, lr in [(32, 0.1, 2e-3), (64, 0.1, 1e-3), (48, 0.2, 2e-3)]],
    "TGT-M": [dict(arch=dict(hidden=h, drop=dr, heads=2), lr=lr, wd=wd)
              for h, dr, lr, wd in [(32, 0.2, 1e-3, 1e-3), (32, 0.3, 5e-4, 1e-3), (48, 0.3, 5e-4, 1e-2)]],
}


def load_chrono(root, kind):
    """Merge <kind>.json and <kind>__A/B.json into one {'model', 'variants'} record."""
    import json
    from pathlib import Path
    out = {"model": kind, "variants": {}}
    for f in sorted(Path(root, "chrono").glob(f"{kind}*.json")):
        if f.stem != kind and not f.stem.startswith(kind + "__"):
            continue
        out["variants"].update(json.loads(f.read_text())["variants"])
    return out
