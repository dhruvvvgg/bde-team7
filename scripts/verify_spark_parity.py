"""Independent pandas oracle; validation only, never called by Spark processing."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def compare(reference, actual, keys, allow_extrema_fix=False, history_reference=None):
    a, b = reference.sort_values(keys).reset_index(drop=True), actual.sort_values(keys).reset_index(drop=True)
    if len(a) != len(b) or set(a.columns) != set(b.columns):
        raise AssertionError(f"shape/schema mismatch: {a.shape}, {b.shape}")
    if not a[keys].astype(str).equals(b[keys].astype(str)):
        raise AssertionError("keys differ")
    changes, corrected = {}, {}
    expected_extrema = {}
    if allow_extrema_fix:
        h = (history_reference if history_reference is not None else a).sort_values(["well_id", "date"]).copy()
        masked = h["gwl_m_bgl"].mask(h["suspect_reason"].fillna("").str.contains("zero_extension"))
        h["_history"] = masked
        for col, method in [("gw_max_to_date_m", "max"), ("gw_min_to_date_m", "min")]:
            h["_expected"] = h.groupby("well_id")["_history"].transform(
                lambda v: getattr(v.shift(1).expanding(min_periods=1), method)())
            expected_extrema[col] = a[keys].merge(h[keys + ["_expected"]], on=keys, validate="one_to_one")["_expected"].to_numpy(float)
    for c in a.columns:
        x, y = a[c], b[c]
        if c == "date":
            same = pd.to_datetime(x).to_numpy(dtype="datetime64[ns]") == pd.to_datetime(y).to_numpy(dtype="datetime64[ns]")
        elif pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x):
            same = np.isclose(x.to_numpy(float), y.to_numpy(float), atol=1e-7, rtol=1e-7, equal_nan=True)
            if allow_extrema_fix and c in ["gw_max_to_date_m", "gw_min_to_date_m"]:
                intended = (x.isna().to_numpy() & y.notna().to_numpy()
                            & np.isclose(y.to_numpy(float), expected_extrema[c], atol=1e-7, rtol=1e-7))
                corrected[c] = int((~same & intended).sum())
                same |= intended
        else:
            same = x.astype(object).where(x.notna(), "<NA>").astype(str).to_numpy() == y.astype(object).where(y.notna(), "<NA>").astype(str).to_numpy()
        if not same.all():
            changes[c] = int((~same).sum())
    return {"unexpected_differences": changes, "documented_extrema_corrections": corrected, "rows": len(a), "columns": len(a.columns)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=Path("processed/spark"))
    p.add_argument("--reference", type=Path, default=Path("processed"))
    args = p.parse_args()
    pairs = [("main", "gw_features_by_state", ["well_id", "date"]),
             ("extension", "extension_2023_2024/extension_features.parquet", ["well_id", "date"]),
             ("splits", "splits.parquet", ["well_id", "period_label"])]
    results = {}
    manifest = json.loads((args.output / "run_manifest.json").read_text())
    has_extension = "extension_rows" in manifest
    for name, relative, keys in pairs:
        if name == "extension" and not has_extension:
            continue
        reference = pd.read_parquet(args.reference / relative)
        if name == "splits" and not has_extension:
            reference = reference[reference.source == "main"]
        history = pd.concat([pd.read_parquet(args.reference / "gw_features_by_state"), reference], ignore_index=True) if name == "extension" else None
        results[name] = compare(reference, pd.read_parquet(args.output / relative), keys,
                                allow_extrema_fix=name != "splits", history_reference=history)
    (args.output / "parity_report.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))
    if any(r["unexpected_differences"] for r in results.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
