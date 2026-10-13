"""Model-ready arrays built ONLY from class-A columns of the preprocessed table.

Leakage rules (see docs/handoff_for_modeling.md):
  * inputs: class A columns, minus lat/lon/state/district and calendar-year indices
    (so a held-out region or future year cannot be recognised by identity);
  * the target is the capped anomaly z of round t; the baseline mean/std used to
    express every depth input relative to the well are the PAST-ONLY baseline of round t;
  * the sequence window holds rounds t-L..t-1 (readings) plus round t with its reading
    hidden (only its season and its rainfall, which ends at month M-1, are visible);
  * scalers are fitted on the training rows only.
"""
from __future__ import annotations

import torch  # noqa: F401  (must load before pandas/pyarrow on Windows)
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from gwstress import contract  # noqa: E402

L = 12  # history rounds in the sequence window (3 years of rounds)
SD_FLOOR = 0.10
DROP = {"lat", "lon", "state", "district", "year", "water_year", "water_year_index",
        "last_monsoon_year", "last_ne_year"}
LAG_COLS = ["gw_prev_round_m", "gw_last_obs_m", "gw_same_season_lag1y_m",
            "gw_same_season_lag2y_m", "gw_max_to_date_m", "gw_min_to_date_m"]
CATS = ["season", "well_type", "aquifer_type"]
SEASONS = ["post_monsoon_rabi", "pre_monsoon", "monsoon", "post_monsoon_kharif"]
CLASSES = ["Normal", "Watch", "Moderate Stress", "High Stress"]


def load_table() -> pd.DataFrame:
    m = pd.read_parquet(ROOT / "processed/gw_features_by_state")
    s = pd.read_parquet(ROOT / "processed/splits.parquet")
    d = m.merge(s[["well_id", "period_label", "state_fold", "split_chrono", "eval_eligible",
                   "no_may_state"]], on=["well_id", "period_label"])
    d["state"] = d["state"].astype(str)
    return d.sort_values(["well_id", "date"]).reset_index(drop=True)


def class_a_numeric(df: pd.DataFrame) -> list[str]:
    cols = []
    for name, cls, kind, *_ in contract.COLUMNS:
        if cls != "A" or name in DROP or name in CATS or name not in df.columns:
            continue
        if kind in ("f", "i", "b"):
            cols.append(name)
    return cols


def build(L: int = L) -> dict:
    d = load_table()
    num = class_a_numeric(d)
    rain_cols = [c for c in num if c.startswith("rain_")]
    bm = d["baseline_mean_m"].to_numpy(float)
    sd = np.maximum(d["baseline_std_m"].to_numpy(float), SD_FLOOR)
    feats = pd.DataFrame({c: d[c].astype(float) for c in num})
    for c in LAG_COLS:  # depth lags re-expressed relative to the well's own baseline
        feats[c + "_rz"] = ((d[c] - d["baseline_mean_m"]) / sd).clip(-10, 10)
    derived = [c + "_rz" for c in LAG_COLS]
    for c in CATS:
        oh = pd.get_dummies(d[c].astype(object).fillna("unknown"), prefix=c).astype(float)
        feats = pd.concat([feats, oh], axis=1)
    rain_set = set(rain_cols)
    names = list(feats.columns)
    is_rain = np.array([c in rain_set for c in names])

    # sequence tensors over the full 92-round long table, gaps kept
    wid = d["well_id"].astype("category").cat.codes.to_numpy()
    n_w = wid.max() + 1
    rounds = d.groupby("well_id").cumcount().to_numpy()
    R = rounds.max() + 1
    idx = np.full((n_w, R), -1, dtype=np.int64)
    idx[wid, rounds] = np.arange(len(d))
    g = d["gwl_m_bgl"].to_numpy(float)
    month = d["month"].to_numpy(int)
    season_id = np.select([month == 1, month == 5, month == 8, month == 11], [0, 1, 2, 3])
    lr1 = np.log1p(d["rain_1m_mm"].to_numpy(float))
    lr3 = np.log1p(d["rain_3m_mm"].to_numpy(float))

    elig = d["eval_eligible"].to_numpy(bool) & d["split_chrono"].isin(["train", "val", "test"]).to_numpy()
    rows = np.flatnonzero(elig)
    w, r = wid[rows], rounds[rows]
    steps = r[:, None] + np.arange(-L, 1)[None, :]            # (n, L+1) round numbers
    valid = steps >= 0
    ridx = np.where(valid, idx[w[:, None], np.clip(steps, 0, R - 1)], -1)
    present = ridx >= 0
    safe = np.where(present, ridx, 0)
    seq = np.zeros((len(rows), L + 1, 10), np.float32)
    val = np.where(present, g[safe], np.nan)
    val[:, -1] = np.nan                                        # current reading hidden
    rel = (val - bm[rows][:, None]) / sd[rows][:, None]
    obs = ~np.isnan(rel)
    seq[:, :, 0] = np.where(obs, np.clip(rel, -10, 10), 0)
    seq[:, :, 1] = obs
    sid = np.where(present, season_id[safe], 0)
    for k in range(4):
        seq[:, :, 2 + k] = (sid == k) & present
    seq[:, :, 6] = present
    seq[:, :, 7] = np.where(present, lr1[safe], 0)
    seq[:, :, 8] = np.where(present, lr3[safe], 0)
    seq[:, :, 9] = 0.0  # reserved

    X = feats.iloc[rows].reset_index(drop=True)
    meta = d.iloc[rows][["well_id", "period_label", "date", "state", "state_fold", "split_chrono",
                         "season", "no_may_state", "suspect_reading", "stress_category",
                         "gwl_m_bgl", "baseline_mean_m", "baseline_std_m", "anomaly_z",
                         "anomaly_z_capped", "gw_prev_round_m", "gw_same_season_lag1y_m",
                         "gw_last_obs_m", "completeness_to_date"]].reset_index(drop=True)
    meta["sd_floor"] = sd[rows]
    return {"X": X, "names": names, "is_rain": is_rain, "meta": meta, "seq": seq,
            "derived": derived, "n_rain_cols": int(is_rain.sum())}


class Scaler:
    """Median-impute + standardise (fit on train rows only). Missing flags for NaN-bearing columns."""

    def fit(self, X: np.ndarray):
        self.med = np.nanmedian(X, 0)
        self.med = np.where(np.isnan(self.med), 0, self.med)
        Xi = np.where(np.isnan(X), self.med, X)
        self.mu, self.sd = Xi.mean(0), Xi.std(0)
        self.sd = np.where(self.sd < 1e-9, 1.0, self.sd)
        self.flag = np.isnan(X).mean(0) > 0.001
        return self

    def transform(self, X: np.ndarray, flags: bool = True) -> np.ndarray:
        miss = np.isnan(X)
        Z = np.clip((np.where(miss, self.med, X) - self.mu) / self.sd, -8, 8).astype(np.float32)
        if flags:
            Z = np.concatenate([Z, miss[:, self.flag].astype(np.float32)], 1)
        return Z


def stress_label(z: np.ndarray) -> np.ndarray:
    """Same thresholds as gwstress.stress.classify (1 / 1.5 / 2 on anomaly z)."""
    return np.select([z >= 2, z >= 1.5, z >= 1], [3, 2, 1], default=0)


def true_label(cat: pd.Series) -> np.ndarray:
    return cat.astype(str).map({c: i for i, c in enumerate(CLASSES)}).to_numpy()


def mask_history(b: dict, p: float, seed: int):
    """Stress test: remove a share p of the observed past readings in the 12-round window (test time only).
    Derived features of removed rounds are set to missing; aggregate history features are left unchanged."""
    X = b["X"].copy()
    seq = b["seq"].copy()
    meta = b["meta"]
    bm, sd = meta["baseline_mean_m"].to_numpy(float), meta["sd_floor"].to_numpy(float)
    rng = np.random.default_rng(seed)
    obs = b["seq"][:, :12, 1] > 0
    drop = obs & (rng.random(obs.shape) < p)
    raw = bm[:, None] + b["seq"][:, :12, 0] * sd[:, None]      # reading value of each window position
    seq[:, :12, 0] = np.where(drop, 0.0, seq[:, :12, 0])
    seq[:, :12, 1] = np.where(drop, 0.0, seq[:, :12, 1])
    alive = obs & ~drop

    def col(name):
        return X[name].to_numpy(float).copy()

    prev, lag1, lag2, chg = col("gw_prev_round_m"), col("gw_same_season_lag1y_m"), col("gw_same_season_lag2y_m"), col("gw_prev_change_m")
    last, rsl = col("gw_last_obs_m"), col("rounds_since_last_obs")
    prev[drop[:, 11]] = np.nan
    lag1[drop[:, 8]] = np.nan
    lag2[drop[:, 4]] = np.nan
    chg[drop[:, 11] | drop[:, 10]] = np.nan
    inside = rsl <= 12                                            # last observation lies in the window
    has = alive.any(1)
    jlast = 11 - np.argmax(alive[:, ::-1], axis=1)                # most recent surviving position
    newlast = np.where(has, raw[np.arange(len(raw)), jlast], np.nan)
    newrsl = np.where(has, 12 - jlast, np.nan)
    chgd = drop.any(1)                                            # rows without a removed reading keep their original values
    last = np.where(inside & chgd, newlast, last)
    rsl = np.where(inside & chgd, newrsl, rsl)
    for name, v in (("gw_prev_round_m", prev), ("gw_same_season_lag1y_m", lag1), ("gw_same_season_lag2y_m", lag2),
                    ("gw_prev_change_m", chg), ("gw_last_obs_m", last), ("rounds_since_last_obs", rsl)):
        X[name] = v
    for name in ("gw_prev_round_m", "gw_last_obs_m", "gw_same_season_lag1y_m", "gw_same_season_lag2y_m"):
        X[name + "_rz"] = ((X[name] - bm) / np.maximum(sd, SD_FLOOR)).clip(-10, 10)
    return X, seq, float(drop.sum() / max(obs.sum(), 1))
