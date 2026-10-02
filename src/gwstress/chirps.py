"""Stage 4a: CHIRPS v3.0 monthly rainfall sampled at every well.

Source: Climate Hazards Center, CHIRPS v3.0 monthly global GeoTIFFs,
https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/
(FTP is blocked in this sandbox; HTTPS only, per checkpoint-1 decision.)

Design decisions (WHY):
- One file at a time, sequential, with backoff: CHC is a public academic
  server; we must not hammer it.
- Raster -> sample -> delete immediately: each global file is ~23 MB and the
  282 needed would be ~6.5 GB; only ~2,759 numbers per file are needed.
  Raw rasters live in dataset/chirps/raw/ which is gitignored.
- Cache after EVERY month: the compact table is rewritten after each file, so
  an interrupted run resumes at the first missing month instead of restarting.
- Nearest cell (0.05 deg ~ 5.5 km): wells are points; bilinear interpolation
  would blend in neighbouring cells across rainfall gradients (e.g. Western
  Ghats) without adding information at this resolution.
- Nodata (CHIRPS uses -9999) and any negative value -> NaN (rainfall cannot be
  negative; ocean / masked cells must not become 0 mm).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config

BASE_URL = "https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/"
FIRST_MONTH = pd.Period("1999-07", "M")   # 6 months before Jan-2000 (longest window)
LAST_MONTH = pd.Period("2022-12", "M")    # last month of the chosen GW table


def months() -> list[pd.Period]:
    return list(pd.period_range(FIRST_MONTH, LAST_MONTH, freq="M"))


def file_name(p: pd.Period) -> str:
    return f"chirps-v3.0.{p.year}.{p.month:02d}.tif"


def download(p: pd.Period, dest_dir: Path, retries: int = 5, timeout: int = 300) -> Path:
    """Download one month with polite exponential backoff (5, 10, 20, 40 s ...)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / file_name(p)
    tmp = dest.with_suffix(".part")
    for attempt in range(retries):
        try:
            with requests.get(BASE_URL + file_name(p), stream=True, timeout=timeout) as r:
                r.raise_for_status()
                expected = int(r.headers.get("content-length", 0))
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if expected and tmp.stat().st_size != expected:
                raise IOError(f"truncated: {tmp.stat().st_size} of {expected} bytes")
            tmp.rename(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - any failure is retried, then re-raised
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            wait = 5 * 2 ** attempt
            print(f"  {file_name(p)} attempt {attempt + 1} failed ({e}); retry in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def sample(tif: Path, wells: pd.DataFrame) -> np.ndarray:
    """Nearest-cell value at each (lat, lon); nodata/negative -> NaN."""
    import rasterio

    with rasterio.open(tif) as ds:
        assert ds.crs is None or ds.crs.to_epsg() == 4326, f"unexpected CRS {ds.crs}"
        # rowcol with the default floor op returns the cell CONTAINING the
        # point, which for a regular grid is the cell whose centre is nearest.
        rows, cols = rasterio.transform.rowcol(ds.transform, wells["lon"].to_numpy(),
                                               wells["lat"].to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        assert (rows >= 0).all() and (rows < ds.height).all() and (cols >= 0).all() and (cols < ds.width).all()
        # Read only the bounding window covering India instead of the whole globe.
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
        arr = ds.read(1, window=win).astype("float64")
        vals = arr[rows - r0, cols - c0]
        nodata = ds.nodata
    if nodata is not None:
        vals[vals == nodata] = np.nan
    vals[vals < 0] = np.nan
    return vals


def build_table(wells: pd.DataFrame, out: Path = config.CHIRPS_TABLE,
                raw_dir: Path = config.CHIRPS_RAW_DIR) -> pd.DataFrame:
    """Resumable download-sample-delete loop. Returns well_id x month long table."""
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(
        columns=["well_id", "year", "month", "precip_mm"])
    have = set(zip(done.year, done.month))
    todo = [p for p in months() if (p.year, p.month) not in have]
    print(f"CHIRPS: {len(have)} months cached, {len(todo)} to fetch", flush=True)
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        tif = download(p, raw_dir)
        try:
            vals = sample(tif, wells)
        finally:
            tif.unlink(missing_ok=True)   # never keep raw rasters on disk
        part = pd.DataFrame({"well_id": wells["well_id"].to_numpy(), "year": p.year,
                             "month": p.month, "precip_mm": vals.astype("float32")})
        done = pd.concat([done, part], ignore_index=True) if len(done) else part
        out.parent.mkdir(parents=True, exist_ok=True)
        done.to_parquet(out, index=False)   # checkpoint after every month
        print(f"  [{i}/{len(todo)}] {p} nan={np.isnan(vals).sum()} "
              f"mean={np.nanmean(vals):.1f}mm {time.time() - t0:.0f}s", flush=True)
        time.sleep(1)  # politeness gap between requests
    return done.sort_values(["well_id", "year", "month"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Part C: sampling check (nearest cell vs 3x3 neighbourhood mean)
# ---------------------------------------------------------------------------
CHIRPS_3X3_TABLE = config.CHIRPS_DIR / "chirps_monthly_by_well_3x3.parquet"


def sample_nearest_and_3x3(tif: Path, wells: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (nearest, mean_3x3, n_valid_3x3) for every well.

    nearest: same rule as sample(): the cell containing the point.
    mean_3x3: mean of that cell and its 8 neighbours. Nodata and negative
        cells are treated as missing and left out of the mean, not counted as
        0 mm. The mean is NaN if all 9 cells are missing.
    WHY 3x3 (~16 x 16 km): checks whether a single 0.05 deg cell is
    representative of the area around the well. Big differences point to
    coastal cells (mixed land/sea), steep rainfall gradients (Western Ghats,
    Himalayan foothills), or raster-edge problems.
    """
    import rasterio

    with rasterio.open(tif) as ds:
        rows, cols = rasterio.transform.rowcol(ds.transform, wells["lon"].to_numpy(),
                                               wells["lat"].to_numpy())
        rows, cols = np.asarray(rows), np.asarray(cols)
        # Window padded by one cell on every side so each 3x3 block is inside it.
        r0, c0 = max(rows.min() - 1, 0), max(cols.min() - 1, 0)
        r1, c1 = min(rows.max() + 2, ds.height), min(cols.max() + 2, ds.width)
        arr = ds.read(1, window=rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)).astype("float64")
        nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    arr[arr < 0] = np.nan
    rr, cc = rows - r0, cols - c0
    nearest = arr[rr, cc]
    # Out-of-bounds neighbours (only at the raster's own edge) count as missing,
    # never as a duplicate of the edge cell.
    padded = np.pad(arr, 1, constant_values=np.nan)
    blocks = np.stack([padded[rr + 1 + dr, cc + 1 + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
    n_valid = (~np.isnan(blocks)).sum(axis=0)
    with np.errstate(invalid="ignore"):
        mean3 = np.nanmean(np.where(n_valid > 0, blocks, 0), axis=0)
    mean3[n_valid == 0] = np.nan
    return nearest, mean3, n_valid


def build_table_3x3(wells: pd.DataFrame, out: Path = CHIRPS_3X3_TABLE,
                    raw_dir: Path = config.CHIRPS_RAW_DIR, last: pd.Period = LAST_MONTH) -> pd.DataFrame:
    """Resumable download, sample and delete loop for the sampling check.
    Same politeness and caching rules as build_table(). Never touches
    CHIRPS_TABLE."""
    cols = ["well_id", "year", "month", "precip_nearest_mm", "precip_3x3_mm", "n_valid_3x3"]
    done = pd.read_parquet(out) if out.exists() else pd.DataFrame(columns=cols)
    have = set(zip(done.year, done.month))
    todo = [p for p in pd.period_range(FIRST_MONTH, last, freq="M") if (p.year, p.month) not in have]
    print(f"CHIRPS 3x3: {len(have)} months cached, {len(todo)} to fetch", flush=True)
    for i, p in enumerate(todo, 1):
        t0 = time.time()
        tif = download(p, raw_dir)
        try:
            near, mean3, nv = sample_nearest_and_3x3(tif, wells)
        finally:
            tif.unlink(missing_ok=True)   # never keep raw rasters on disk
        part = pd.DataFrame({"well_id": wells["well_id"].to_numpy(), "year": p.year, "month": p.month,
                             "precip_nearest_mm": near.astype("float32"),
                             "precip_3x3_mm": mean3.astype("float32"), "n_valid_3x3": nv.astype("int8")})
        done = pd.concat([done, part], ignore_index=True) if len(done) else part
        done.to_parquet(out, index=False)
        print(f"  [{i}/{len(todo)}] {p} nan_near={np.isnan(near).sum()} "
              f"partial3x3={(nv < 9).sum()} {time.time() - t0:.0f}s", flush=True)
        time.sleep(1)
    return done.sort_values(["well_id", "year", "month"]).reset_index(drop=True)
