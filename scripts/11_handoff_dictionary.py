"""Part H: write the column dictionary in docs/handoff_for_modeling.md from
src/gwstress/contract.py (between the BEGIN/END GENERATED markers), so the docs
always match the data contract that tests/test_data_contract.py enforces.

Run: python scripts/11_handoff_dictionary.py
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, contract  # noqa: E402

DOC = config.REPO_ROOT / "docs" / "handoff_for_modeling.md"
KIND = {"f": "float", "i": "int", "b": "bool", "s": "string", "c": "category", "M": "datetime"}
GROUPS = [("Keys and calendar", ["well_id", "date", "period_label", "year", "month", "season", "water_year",
                                  "water_year_index", "season_order"]),
          ("Reading (target)", ["gwl_m_bgl"]),
          ("Static covariates", ["state", "state_slug", "district", "station_name", "lat", "lon", "well_type",
                                 "aquifer_type", "well_depth_m", "specific_yield"]),
          ("Groundwater history (Part A)", None),
          ("Rainfall (stage 4, Parts A, A2)", None),
          ("Baselines, anomalies, labels (stage 5-6, Parts B, B2)", None),
          ("Trend and completeness", ["trend_5y_m_per_yr", "completeness_to_date", "well_completeness_full_period",
                                      "well_has_no_series_for_season"]),
          ("QC flags (stage 3, Part D)", None),
          ("Extension only", ["is_extension"])]


def group_of(name: str) -> str:
    for g, cols in GROUPS:
        if cols and name in cols:
            return g
    if name.startswith("gw_") or name in ("n_readings_to_date", "rounds_since_last_obs"):
        return "Groundwater history (Part A)"
    if name.startswith("rain_") or name in ("last_monsoon_year", "last_ne_year"):
        return "Rainfall (stage 4, Parts A, A2)"
    if name.startswith("flag_") and "std_floored" not in name or name.startswith("suspect_"):
        return "QC flags (stage 3, Part D)"
    return "Baselines, anomalies, labels (stage 5-6, Parts B, B2)"


def render() -> str:
    rows = contract.COLUMNS + contract.EXTENSION_ONLY
    out = [f"{len(contract.COLUMNS)} columns in the main table (+ `is_extension` in the extension table). "
           f"Class counts: " + ", ".join(f"{k}: {sum(r[1] == k for r in contract.COLUMNS)}" for k in "ABC") + ".\n"]
    for g, _ in GROUPS:
        sel = [r for r in rows if group_of(r[0]) == g]
        if not sel:
            continue
        out += [f"\n**{g}**\n", "| column | class | type | NaN when | range | description |",
                "|---|:-:|---|---|---|---|"]
        for name, cls, kind, nan, lo, hi, desc in sel:
            rng = "" if lo is None else f"{lo:g} to {hi:g}".replace("inf", "∞")
            out.append(f"| `{name}` | {cls} | {KIND[kind]} | {'never' if nan == 'none' else nan} | {rng} | {desc} |")
    return "\n".join(out)


def main() -> None:
    text = DOC.read_text()
    pat = re.compile(r"(<!-- BEGIN GENERATED COLUMN DICTIONARY[^\n]*-->\n)(.*?)(<!-- END GENERATED COLUMN DICTIONARY -->)", re.S)
    assert pat.search(text), "markers not found in handoff doc"
    DOC.write_text(pat.sub(lambda m: m.group(1) + render() + "\n" + m.group(3), text))
    print(f"wrote {len(contract.COLUMNS) + 1} column entries to {DOC.relative_to(config.REPO_ROOT)}")


if __name__ == "__main__":
    main()
