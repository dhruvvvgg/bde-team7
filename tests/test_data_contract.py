"""Data contract tests for the committed outputs (Part G).

The expected columns, dtypes, ranges and NaN policies come from
src/gwstress/contract.py, the same module that generates the column dictionary
in docs/handoff_for_modeling.md.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import chirps, config, contract, stress  # noqa: E402

EXT_PATH = config.EXT_DIR / "extension_features.parquet"
SPLITS_PATH = config.PROCESSED_DIR / "splits.parquet"
pytestmark = pytest.mark.skipif(not config.FINAL_DIR.exists(), reason="final table not built")

KIND = {"f": "f", "i": "i", "b": "b", "M": "M"}


@pytest.fixture(scope="module")
def main():
    return pd.read_parquet(config.FINAL_DIR)


@pytest.fixture(scope="module")
def ext():
    return pd.read_parquet(EXT_PATH)


def _check_against_contract(df, specs):
    errors = []
    for name, cls, kind, nan_policy, lo, hi, _ in specs:
        s = df[name]
        if kind in KIND:
            if s.dtype.kind != KIND[kind]:
                errors.append(f"{name}: dtype {s.dtype} != kind {kind}")
        elif kind == "c":
            if not isinstance(s.dtype, pd.CategoricalDtype):
                errors.append(f"{name}: not categorical")
        elif kind == "s":
            if s.dtype.kind not in "OU" and str(s.dtype) not in ("str", "string"):
                errors.append(f"{name}: not string ({s.dtype})")
        if nan_policy == "none" and s.isna().any():
            errors.append(f"{name}: {int(s.isna().sum())} NaN but policy is 'none'")
        if lo is not None and kind in "fi":
            v = s.dropna().to_numpy(float)
            tol = 1e-9 * max(1.0, abs(lo) if np.isfinite(lo) else 1.0)   # float rounding only
            if len(v) and (v.min() < lo - tol or v.max() > hi + tol):
                errors.append(f"{name}: range [{v.min()}, {v.max()}] outside [{lo}, {hi}]")
    assert not errors, "\n".join(errors)


def test_main_columns_match_contract_exactly(main):
    assert set(main.columns) == {c[0] for c in contract.COLUMNS}


def test_main_dtypes_ranges_nan_policy(main):
    _check_against_contract(main, contract.COLUMNS)


def test_main_grid_and_key_uniqueness(main):
    assert not main.duplicated(contract.KEY).any()
    assert not main.duplicated(["well_id", "period_label"]).any()
    assert main.well_id.nunique() == 2759 and main.date.nunique() == 92
    assert len(main) == 2759 * 92
    assert main.groupby("well_id").size().eq(92).all()
    assert main.year.between(2000, 2022).all()


def test_no_nan_in_key_and_rainfall_columns(main):
    assert main[contract.KEY + ["period_label", "state", "season"]].notna().all().all()
    assert main[contract.NO_NAN_RAINFALL].notna().all().all()
    # Documented exceptions: CHIRPS starts 1999-07, so these windows are incomplete
    # only for the rounds listed, and nowhere else.
    nan12 = main[main.rain_12m_mm.isna()].period_label.unique()
    assert set(nan12) == {"Jan-00", "May-00"}
    assert set(main[main.rain_monsoon_ytd_mm.isna()].period_label.unique()) == {"Jan-00", "May-00"}
    assert set(main[main.rain_last_monsoon_mm.isna()].period_label.unique()) == {"Jan-00", "May-00", "Aug-00"}


def test_categorical_values(main):
    for c in ["stress_category", "stress_roll10", "stress_tadj"]:
        assert set(main[c].astype(str).unique()) <= set(stress.CATEGORIES)
        assert (main[c].astype(str) == stress.NO_READING).sum() == main.gwl_m_bgl.isna().sum()
    assert set(main.season.astype(str).unique()) == set(config.SEASON_BY_MONTH.values())
    reasons = main.suspect_reason.dropna().str.split(";").explode()
    assert set(reasons) <= {"zero_suspicious", "deeper_than_well", "extreme_anomaly"}
    assert (main.suspect_reason.notna() == main.suspect_reading).all()


def test_partitioned_parquet_reads_back_identical(main, tmp_path):
    # 1. Each partition directory holds exactly one state, matching its slug.
    for d in config.FINAL_DIR.iterdir():
        part = pd.read_parquet(d)
        assert part.state.nunique() == 1
        assert d.name == "state_slug=" + part.state.iloc[0].lower().replace(" ", "_")
    # 2. Round trip: write with the same partitioning and read back unchanged.
    out = tmp_path / "rt"
    main.drop(columns="state_slug").assign(state_slug=main.state_slug.astype(str)) \
        .to_parquet(out, partition_cols=["state_slug"], index=False)
    back = pd.read_parquet(out)
    key = contract.KEY
    a = main.sort_values(key).reset_index(drop=True)
    b = back.sort_values(key).reset_index(drop=True)[a.columns]
    b["state_slug"] = b.state_slug.astype(str)
    a = a.assign(state_slug=a.state_slug.astype(str))
    pd.testing.assert_frame_equal(a, b, check_categorical=False, check_dtype=False)


def test_extension_contract(ext, main):
    specs = [c if c[0] != "year" else ("year", "A", "i", "none", 2023, 2024, "") for c in contract.COLUMNS]
    specs = [c for c in specs if c[0] != "state_slug"]
    assert set(ext.columns) == {c[0] for c in specs} | {"is_extension"}
    _check_against_contract(ext, specs + contract.EXTENSION_ONLY)
    assert len(ext) == 2759 * 8 and not ext.duplicated(contract.KEY).any()
    assert set(ext.well_id) == set(main.well_id)
    assert not set(zip(ext.well_id, ext.date)) & set(zip(main.well_id, main.date))
    # static covariates identical to the main table
    stat = ["state", "district", "lat", "lon", "well_type", "well_depth_m", "specific_yield"]
    m1 = main.drop_duplicates("well_id").set_index("well_id")[stat]
    e1 = ext.drop_duplicates("well_id").set_index("well_id")[stat].loc[m1.index]
    pd.testing.assert_frame_equal(m1, e1, check_dtype=False)


def test_splits_join_one_to_one(main, ext):
    s = pd.read_parquet(SPLITS_PATH)
    assert not s.duplicated(["well_id", "period_label"]).any()
    sm = s[s.source == "main"]
    j = main[["well_id", "period_label"]].merge(sm, on=["well_id", "period_label"], how="outer", indicator=True)
    assert (j._merge == "both").all() and len(j) == len(main)
    se = s[s.source == "extension"]
    j = ext[["well_id", "period_label"]].merge(se, on=["well_id", "period_label"], how="outer", indicator=True)
    assert (j._merge == "both").all() and len(j) == len(ext)
    # eligibility is consistent with the main table's own columns
    m = main.merge(sm, on=["well_id", "period_label"])
    assert not (m.eval_eligible & m.gwl_m_bgl.isna()).any()
    assert not (m.eval_eligible & m.anomaly_z.isna()).any()


def test_well_level_files_join(main):
    cw = pd.read_csv(config.CROSSWALK)
    assert cw.well_id.is_unique and set(cw.well_id) == set(main.well_id)
    for path, n_months in [(config.CHIRPS_TABLE, len(chirps.months())), (chirps.CHIRPS_3X3_TABLE, len(chirps.months())),
                           (config.CHIRPS_EXT_TABLE, 24)]:
        t = pd.read_parquet(path)
        assert set(t.well_id) == set(main.well_id), path.name
        assert not t.duplicated(["well_id", "year", "month"]).any() and len(t) == 2759 * n_months, path.name
    r1 = pd.read_parquet(config.CHIRPS_TABLE)
    r3 = pd.read_parquet(chirps.CHIRPS_3X3_TABLE)
    j = r1.merge(r3, on=["well_id", "year", "month"], how="outer", indicator=True)
    assert (j._merge == "both").all()
    np.testing.assert_allclose(j.precip_mm, j.precip_nearest_mm)


def test_every_committed_file_under_50mb():
    root = config.REPO_ROOT
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True).stdout.split()
    big = [f for f in files if (root / f).exists() and (root / f).stat().st_size > 50e6]
    assert not big, big
