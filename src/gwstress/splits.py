"""Part F: shared evaluation splits, keyed by (well_id, period_label).

All cutoffs come from config/splits.json; regenerate with scripts/10_splits.py.

eval_eligible = reading exists
                AND baseline exists under K (n_prior_same_season >= K, so anomaly_z is not NaN)
                AND the round is not sparse (coverage below the config threshold).
exclusion_reason: first failing condition, in that order: 'no_reading',
                  'extension_zero_placeholder' (an extension reading flagged
                  zero_extension), 'insufficient_history', 'sparse_round'.
                  NaN if eligible.
split_chrono:  'train' / 'val' / 'test' / 'ext' by year (config "chrono").
               It is assigned to EVERY row, so rows that are not eligible keep
               their period. Filter on eval_eligible to train or evaluate.
state_fold:    integer 0..18, states in alphabetical order (stable across runs).
no_may_state:  per-well flag, True if the well has no pre-monsoon (May)
               reading in 2000-2022 (the 444 wells). Named as requested; it is
               a WELL property (every Kerala and West Bengal well, all but 2
               in Odisha, all of Assam).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import config

CONFIG_PATH = config.REPO_ROOT / "config" / "splits.json"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text())


def build(main: pd.DataFrame, ext: pd.DataFrame | None, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["well_id", "date", "period_label", "year", "season", "state", "gwl_m_bgl", "n_prior_same_season",
            "suspect_reason"]
    parts = [main[cols].assign(source="main")]
    if ext is not None:
        parts.append(ext[cols].assign(source="extension"))
    df = pd.concat(parts, ignore_index=True)
    df["season"] = df["season"].astype(str)

    cov = (df.groupby(["source", "period_label"]).gwl_m_bgl
           .apply(lambda s: 100 * s.notna().mean()).rename("coverage_pct").reset_index())
    thr = np.where(cov.source == "main", cfg["sparse_round_min_coverage_pct"],
                   cfg["ext_sparse_round_min_coverage_pct"])
    cov["is_sparse"] = cov.coverage_pct < thr
    df = df.merge(cov[["source", "period_label", "is_sparse"]], on=["source", "period_label"], how="left")

    has = df.gwl_m_bgl.notna()
    hist = df.n_prior_same_season >= cfg["K_min_history"]
    # Extension zeros (likely placeholders, clustered in the dry May-23 round)
    # are never evaluated. The main table's 11 zeros are NOT excluded: they sit
    # in monsoon rounds where a full well is plausible, and 3 are flagged as
    # suspect for sensitivity runs only.
    ext_zero = (df.source == "extension") & df.suspect_reason.fillna("").str.contains("zero_extension")
    df["exclusion_reason"] = np.select(
        [~has, ext_zero, ~hist, df["is_sparse"].astype(bool)],
        ["no_reading", "extension_zero_placeholder", "insufficient_history", "sparse_round"], default=None)
    df["eval_eligible"] = df.exclusion_reason.isna()

    df["split_chrono"] = None
    for name, (lo, hi) in cfg["chrono"].items():
        df.loc[df.year.between(lo, hi), "split_chrono"] = name
    assert df.split_chrono.notna().all(), "a year is not covered by any chrono split"

    states = sorted(df.state.unique())
    df["state_fold"] = df.state.map({s: i for i, s in enumerate(states)}).astype("int8")
    no_may = (df[(df.source == "main") & (df.season == "pre_monsoon")]
              .groupby("well_id").gwl_m_bgl.apply(lambda s: s.notna().sum() == 0))
    df["no_may_state"] = df.well_id.map(no_may).astype(bool)

    out = df[["well_id", "period_label", "date", "year", "season", "source", "state", "state_fold",
              "split_chrono", "eval_eligible", "exclusion_reason", "no_may_state"]]
    assert not out.duplicated(["well_id", "period_label"]).any()
    return out.sort_values(["well_id", "date"]).reset_index(drop=True), cov
