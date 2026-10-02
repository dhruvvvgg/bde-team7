"""Loading the authors' groundwater CSVs.

Layout (verified in stage 1, not assumed): WIDE. One row per well, 20
metadata columns, then one column per CGWB monitoring round labelled
"Mon-YY" (Jan/May/Aug/Nov, 2000..end year). The ``*_ref_sy_*`` files add a
trailing ``Reference_Sy`` column.

Known upstream defect: ``Station Code`` was saved through a spreadsheet and is
stored in scientific notation (e.g. ``1.4113E+14``), which destroyed digits.
In the 2000-2022 refined table only 2,411 of 2,759 codes are unique, and in
the ``ref_sy`` copy (rounded further, e.g. ``1.41E+14``) only 182 are unique.
It therefore cannot be used as a key. We build a surrogate ``well_id`` from
attributes that ARE unique per row (see make_well_id).
"""
from __future__ import annotations

import hashlib
import re

import pandas as pd

from . import config

OBS_COL_RE = re.compile(r"^(Jan|May|Aug|Nov)-(\d{2})$")

# Attributes that jointly identify a well. Verified unique across all 2,759
# rows of the refined table; kept deliberately to columns that do not change
# between the authors' files so the same well gets the same id in every file.
WELL_KEY_COLS = ["State", "District", "Station Name", "Latitude", "Longitude"]


def split_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (metadata_cols, observation_cols, static_extra_cols).

    WHY parse instead of slicing at 20: the Sy file appends a column after the
    observations, so a positional split would misclassify it as a reading.
    """
    obs = [c for c in df.columns if OBS_COL_RE.match(c)]
    first_obs = df.columns.get_loc(obs[0])
    meta = [c for c in df.columns[:first_obs]]
    extra = [c for c in df.columns[first_obs:] if c not in obs]
    return meta, obs, extra


def parse_obs_label(label: str) -> tuple[int, int, str]:
    """'May-07' -> (2007, 5, 'pre_monsoon'). All data is 2000+, so YY -> 20YY."""
    m = OBS_COL_RE.match(label)
    if not m:
        raise ValueError(f"not an observation label: {label!r}")
    mon, yy = m.groups()
    return 2000 + int(yy), config.MONTH_NUM[mon], config.SEASON_BY_MONTH[mon]


def make_well_id(df: pd.DataFrame, strict: bool = True) -> pd.Series:
    """Deterministic surrogate key: 'W' + 10 hex chars of SHA-1 over WELL_KEY_COLS.

    WHY a hash rather than the row number: row order can differ between the
    authors' files; the hash is reproducible from content alone. Coordinates
    are formatted with 6 decimals so float repr noise cannot change the id.
    """
    def _key(r) -> str:
        parts = [str(r["State"]).strip().lower(), str(r["District"]).strip().lower(),
                 str(r["Station Name"]).strip().lower(),
                 f"{float(r['Latitude']):.6f}", f"{float(r['Longitude']):.6f}"]
        return "W" + hashlib.sha1("|".join(parts).encode()).hexdigest()[:10]

    ids = df.apply(_key, axis=1)
    # strict=False is only for profiling the authors' intermediate tables
    # (stages 2-4), which contain 149 colliding rows; the refined 2000-YYYY
    # tables have none, and the pipeline always loads with strict=True.
    if strict and ids.duplicated().any():
        raise ValueError(f"{ids.duplicated().sum()} well_id collisions; key is not unique")
    return ids


def load_wide(path=config.GW_REFINED_FILE, strict: bool = True) -> pd.DataFrame:
    """Read a wide groundwater CSV unchanged and prepend ``well_id``.

    Station Code is read as string so we never silently re-round it further.
    """
    df = pd.read_csv(path, dtype={"Station Code": str}, low_memory=False)
    ids = make_well_id(df, strict=strict)
    # concat instead of insert: avoids pandas' fragmented-frame warning on wide data
    return pd.concat([ids.rename("well_id"), df], axis=1)
