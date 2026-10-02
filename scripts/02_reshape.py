"""Stage 2: reshape the refined wide table to long and write the crosswalk.

Run: python scripts/02_reshape.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import config, gw_io, reshape  # noqa: E402


def main() -> None:
    wide = gw_io.load_wide(config.GW_REFINED_FILE)  # strict: raises on id collision
    cw = reshape.crosswalk(wide)
    # The Sy file's Station Code is rounded to 3 significant digits (1.41E+14);
    # the same wells in Filtered_GWLs_2000_2024/..._2000_2022.csv keep 5 digits
    # (1.4113E+14). Both are corrupted; we keep both so the better one is usable.
    base = gw_io.load_wide(config.GW_FILTERED_DIR / "CGWB_filtered_wells_2000_2022.csv")
    codes = base.set_index("well_id")["Station Code"]
    assert set(codes.index) == set(cw.well_id), "Sy file and 2000_2022 file disagree on wells"
    cw["station_code_corrupted_5sig"] = cw.well_id.map(codes)
    cw.to_csv(config.CROSSWALK, index=False)
    long = reshape.to_long(wide)
    # Integrity: every well x every round exactly once.
    n_rounds = long.period_label.nunique()
    assert len(long) == len(wide) * n_rounds, "melt lost or duplicated rows"
    assert long.gwl_m_bgl.notna().sum() == wide.filter(regex=r"^(Jan|May|Aug|Nov)-\d\d$").notna().sum().sum()
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    long.to_parquet(config.GW_LONG, index=False)
    print(f"wells={long.well_id.nunique()} rounds={n_rounds} rows={len(long)} "
          f"readings={long.gwl_m_bgl.notna().sum()} -> {config.GW_LONG}")
    print(long.head(3).T)


if __name__ == "__main__":
    main()
