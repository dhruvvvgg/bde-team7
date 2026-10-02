"""Stage 6: groundwater stress classification from the seasonal anomaly.

Direction: gwl is depth below ground, so a POSITIVE anomaly (deeper than this
well's usual level for the season) means stress. A negative anomaly
(shallower than usual, e.g. after a wet monsoon) is "Normal", not a separate
category: the project question is stress, and surplus is not stress.

Categories, in precedence order:
  1. "No reading"           - the round has no measurement (missing data, not history)
  2. "Insufficient history" - reading exists, but fewer than K prior same-season
                              readings, so no baseline (anomaly is NaN)
  3. by anomaly z with thresholds (t_watch, t_moderate, t_high), default (1, 1.5, 2):
       z <  t_watch               -> "Normal"
       t_watch    <= z < t_moderate -> "Watch"
       t_moderate <= z < t_high     -> "Moderate Stress"
       z >= t_high                  -> "High Stress"
WHY thresholds are parameters: SD cut-offs are a convention, not a physical
law; the report shows how category counts move under alternative sets.
WHY raw vs capped z does not matter here: every default threshold is < 5
(the cap), so both give identical categories (asserted in classify()).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

NO_READING = "No reading"
INSUFFICIENT = "Insufficient history"
CATEGORIES = [NO_READING, INSUFFICIENT, "Normal", "Watch", "Moderate Stress", "High Stress"]
DEFAULT_THRESHOLDS = (1.0, 1.5, 2.0)

# Threshold sets used in the sensitivity table (name -> thresholds).
SENSITIVITY_SETS = {
    "lenient (0.5/1/1.5)": (0.5, 1.0, 1.5),
    "default (1/1.5/2)": DEFAULT_THRESHOLDS,
    "strict (1.5/2/2.5)": (1.5, 2.0, 2.5),
    "wide (1/2/3)": (1.0, 2.0, 3.0),
}


def classify(gwl: pd.Series, anomaly_z: pd.Series,
             thresholds: tuple[float, float, float] = DEFAULT_THRESHOLDS) -> pd.Series:
    """Return an ordered categorical stress label per row."""
    t1, t2, t3 = thresholds
    if not (t1 < t2 < t3):
        raise ValueError(f"thresholds must be strictly increasing, got {thresholds}")
    z = anomaly_z.to_numpy(dtype=float)
    lab = np.select(
        [gwl.isna().to_numpy(), np.isnan(z), z >= t3, z >= t2, z >= t1],
        [NO_READING, INSUFFICIENT, "High Stress", "Moderate Stress", "Watch"],
        default="Normal")
    return pd.Series(pd.Categorical(lab, categories=CATEGORIES, ordered=True), index=gwl.index)


def sensitivity(df: pd.DataFrame, sets=SENSITIVITY_SETS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(category counts per threshold set, rows changing category vs default)."""
    labs = {name: classify(df["gwl_m_bgl"], df["anomaly_z"], th) for name, th in sets.items()}
    counts = pd.DataFrame({n: l.value_counts().reindex(CATEGORIES) for n, l in labs.items()})
    base = labs["default (1/1.5/2)"]
    classified = ~base.isin([NO_READING, INSUFFICIENT])
    changes = pd.DataFrame([{
        "threshold set": n, "rows changed vs default": int((l != base).sum()),
        "% of classified rows": round(100 * (l != base)[classified].mean(), 2),
        "moved to more severe": int((l.cat.codes > base.cat.codes).sum()),
        "moved to less severe": int((l.cat.codes < base.cat.codes).sum()),
    } for n, l in labs.items()])
    return counts, changes
