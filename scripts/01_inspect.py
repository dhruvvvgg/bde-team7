"""Stage 1: inspection of the groundwater dataset. Read-only; writes
reports/stage1_inspection.md.

Run:  python scripts/01_inspect.py

WHY a script that writes a report (rather than an ad-hoc notebook): the
numbers quoted in the written report must be regenerable from the committed
data with one command.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, gw_io  # noqa: E402

out: list[str] = []


def w(s: str = "") -> None:
    out.append(s)


def md(df: pd.DataFrame, **kw) -> str:
    return df.to_markdown(**kw) if hasattr(df, "to_markdown") else df.to_string()


def inventory() -> None:
    w("## 1. File inventory (dataset/groundwater_data/)\n")
    rows = []
    for p in sorted(config.GW_DIR.rglob("*")):
        if p.is_file():
            rows.append((str(p.relative_to(config.GW_DIR)), p.stat().st_size))
    w("| file | bytes |\n|---|---:|")
    for f, s in rows:
        w(f"| {f} | {s:,} |")
    w(f"\n`Hydrogeological_map/` present: **{(config.GW_DIR / 'Hydrogeological_map').exists()}**\n")


def candidates() -> None:
    w("## 2. Candidate tables\n")
    files = sorted(config.GW_FILTERED_DIR.glob("*.csv")) + [
        config.GW_OUTPUT_DIR / "CGWB_India_filtered_GWLs_ref_sy_2000_2022.csv",
        config.GW_OUTPUT_DIR / "CGWB_India_filtered_Dug_wells_GWLs_ref_sy_2000_2022.csv",
        config.GW_OUTPUT_DIR / "4_India_GWLs_2000_2024_after_3sigma.csv",
    ]
    w("| file | rows | unique Station Code | unique well_id | states | districts | first | last | obs cols | cell missing % | extra cols |")
    w("|---|---:|---:|---:|---:|---:|---|---|---:|---:|---|")
    for f in files:
        d = gw_io.load_wide(f, strict=False)
        meta, obs, extra = gw_io.split_columns(d.drop(columns="well_id"))
        w(f"| {f.name} | {len(d)} | {d['Station Code'].nunique()} | {d.well_id.nunique()} | "
          f"{d.State.nunique()} | {d.District.nunique()} | {obs[0]} | {obs[-1]} | {len(obs)} | "
          f"{100 * d[obs].isna().to_numpy().mean():.1f} | {', '.join(extra) or '-'} |")
    w()


def profile(df: pd.DataFrame) -> None:
    meta, obs, extra = gw_io.split_columns(df.drop(columns="well_id"))
    w(f"## 3. Proposed table: `{config.GW_REFINED_FILE.name}`\n")
    w(f"Shape: **{df.shape}** (incl. surrogate `well_id`).\n")
    w(f"- Metadata columns ({len(meta)}): {', '.join(meta)}")
    w(f"- Observation columns ({len(obs)}): {obs[0]} ... {obs[-1]}")
    w(f"- Static extra columns: {', '.join(extra)}\n")

    w("### dtypes (metadata + extras)\n")
    w(md(df[meta + extra].dtypes.astype(str).to_frame("dtype")))
    w("\nAll observation columns are float64: "
      f"{all(str(df[c].dtype) == 'float64' for c in obs)}\n")

    w("### Sample rows (first 3, selected columns)\n")
    w(md(df[["well_id", "Station Code", "Station Name", "State", "District",
             "Latitude", "Longitude", "Type of Well", "Well Depth"] + obs[:4] + extra].head(3), index=False))

    w("\n### Metadata value counts\n")
    for c in ["Station Type", "Agency Name", "Data Acquisition", "Type of Well", "Aquifer Type",
              "Data Type Code", "Data Type Description", "Unit"] + extra:
        vc = df[c].value_counts(dropna=False)
        w(f"- **{c}**: " + ", ".join(f"{k}={v}" for k, v in vc.items()))
    w()
    w("### Numeric metadata\n")
    w(md(df[["Latitude", "Longitude", "Well Depth"]].describe().T.round(3)))

    long = df.melt(id_vars="well_id", value_vars=obs, var_name="label", value_name="gwl")
    parsed = long.label.map(gw_io.parse_obs_label)
    long["year"] = [p[0] for p in parsed]
    long["month"] = [p[1] for p in parsed]
    long["season"] = [p[2] for p in parsed]

    v = long.gwl.dropna()
    w("\n### Groundwater level (all non-null readings, metres below ground level)\n")
    w(md(v.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(3).to_frame("gwl")))
    w(f"\n- exact zeros: **{(v == 0).sum()}**; negatives: **{(v < 0).sum()}**; "
      f"<= -999 placeholders: **{(v <= -999).sum()}**; > 100 m: **{(v > 100).sum()}**\n")

    w("## 4. Missingness\n")
    w(f"Overall cell missingness (wells x {len(obs)} rounds): "
      f"**{100 * long.gwl.isna().mean():.2f}%** "
      f"({long.gwl.isna().sum():,} of {len(long):,} cells; {long.gwl.notna().sum():,} readings)\n")
    w("### Per season\n")
    ps = long.groupby(["month", "season"]).gwl.agg(n="size", readings="count")
    ps["missing_%"] = (100 * (1 - ps.readings / ps.n)).round(2)
    w(md(ps))
    w("\n### Per year\n")
    py = long.groupby("year").gwl.agg(n="size", readings="count")
    py["missing_%"] = (100 * (1 - py.readings / py.n)).round(2)
    py["wells_with_any"] = long.dropna(subset=["gwl"]).groupby("year").well_id.nunique()
    w(md(py))
    w("\n### Per year x season (missing %)\n")
    t = long.pivot_table(index="year", columns="season", values="gwl",
                         aggfunc=lambda s: round(100 * s.isna().mean(), 1))
    w(md(t[list(dict.fromkeys(config.SEASON_BY_MONTH.values()))]))

    w("\n## 5. Readings per well and per well-season\n")
    rpw = long.groupby("well_id").gwl.count()
    w(f"Readings per well (max possible {len(obs)}):\n")
    w(md(rpw.describe(percentiles=[.01, .05, .1, .25, .5, .75, .9]).round(1).to_frame("readings")))
    bins = [0, 30, 40, 50, 60, 70, 80, 90, len(obs)]
    w("\nDistribution:\n")
    w(md(pd.cut(rpw, bins, include_lowest=True).value_counts().sort_index().to_frame("wells")))
    n_years = long.year.nunique()
    rpws = long.groupby(["well_id", "season"]).gwl.count().unstack()
    w(f"\nReadings per well-season (max possible {n_years} per season):\n")
    w(md(rpws.describe(percentiles=[.05, .1, .25, .5, .75]).round(1)))
    w("\nWells by minimum same-season count across the 4 seasons (relevant for the "
      "minimum-history rule K in stage 5):\n")
    m = rpws.min(axis=1)
    w(md(pd.DataFrame({k: [(m >= k).sum()] for k in [3, 5, 8, 10, 15, 20]}, index=["wells_with_>=k_in_every_season"])))
    full_years = long.groupby(["well_id", "year"]).gwl.count().eq(4).groupby("well_id").sum()
    w(f"\nWells with all 4 rounds filled in **every** year: **{(full_years == n_years).sum()}** "
      f"(median full years per well: {full_years.median():.0f} of {n_years}).")
    zero_pre = rpws["pre_monsoon"].eq(0)
    st = df.set_index("well_id").loc[zero_pre[zero_pre].index, "State"].value_counts()
    w(f"\nWells with **zero** pre-monsoon (May) readings in all {n_years} years: **{zero_pre.sum()}** "
      "(" + ", ".join(f"{k}: {v}" for k, v in st.items()) + "). Every Kerala and West Bengal well, "
      "and nearly every Odisha and Assam well, has no May series.")
    two_per_year = long.groupby(["well_id", "year"]).gwl.count().ge(2).groupby("well_id").all()
    w(f"\nWells with >=2 readings in every year (authors' criterion 4): "
      f"**{two_per_year.sum()} / {len(two_per_year)}**.")

    w("\n## 6. Static columns\n")
    sy = df["Reference_Sy"]
    w(f"- `Reference_Sy`: non-null {sy.notna().sum()}/{len(sy)}; values "
      + ", ".join(f"{k}: {v}" for k, v in sy.value_counts().sort_index().items()))
    w("  Assigned per well by the authors from k-means clusters of the hydrogeological map's RGB "
      "colour at the well location (see Code/.ipynb_checkpoints/2_specific_yield_extraction-checkpoint.ipynb). "
      "It is therefore a 5-level categorical proxy for aquifer class, not a measured value.")
    ct = pd.crosstab(df["Aquifer Type"], sy)
    w("\nAquifer Type x Reference_Sy:\n")
    w(md(ct))
    w("\n- Other static: Type of Well, Aquifer Type, Well Depth, Latitude, Longitude, State, District, Tehsil, Block, Village.")


def main() -> None:
    w("# Stage 1 inspection report\n")
    w("_Generated by `scripts/01_inspect.py`. Do not edit by hand._\n")
    inventory()
    candidates()
    profile(gw_io.load_wide(config.GW_REFINED_FILE))
    config.REPORTS_DIR.mkdir(exist_ok=True)
    (config.REPORTS_DIR / "stage1_inspection.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
