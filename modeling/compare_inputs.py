"""Which model inputs differ between the pandas reference table and the Spark table?

Builds the model arrays (data.build) once from each table and compares every input column, the sequence tensor and the target/meta
columns. Writes results_spark/INPUT_DIFF.md (summary + the first 80 differing rows of each extrema column).

    python modeling/compare_inputs.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCR = Path(os.environ.get("BDE_SCRATCH", ROOT / "results_spark" / "_scratch"))


def dump(tag, processed):
    code = ("import sys, numpy as np; sys.path.insert(0, 'modeling'); import data as D; b = D.build(); "
            f"np.savez_compressed(r'{SCR}/{tag}.npz', X=b['X'].to_numpy(float), seq=b['seq'], names=np.array(b['names']), "
            "y=b['meta']['anomaly_z_capped'].to_numpy(float), well=b['meta']['well_id'].to_numpy(str), "
            "period=b['meta']['period_label'].to_numpy(str), split=b['meta']['split_chrono'].to_numpy(str), "
            "label=b['meta']['stress_category'].to_numpy(str), baseline=b['meta']['baseline_mean_m'].to_numpy(float))")
    env = dict(os.environ, BDE_PROCESSED=str(processed))
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)


def main():
    SCR.mkdir(parents=True, exist_ok=True)
    dump("pandas", ROOT / "processed")
    dump("spark", ROOT / "processed" / "spark")
    a, b = np.load(SCR / "pandas.npz"), np.load(SCR / "spark.npz")
    assert (a["well"] == b["well"]).all() and (a["period"] == b["period"]).all(), "row order differs"
    names = list(a["names"])
    assert names == list(b["names"])
    Xa, Xb = a["X"], b["X"]
    n_rows = len(Xa)
    diff_cols = {}
    for j, c in enumerate(names):
        same = np.isclose(Xa[:, j], Xb[:, j], atol=1e-7, rtol=1e-7, equal_nan=True)
        if not same.all():
            diff_cols[c] = int((~same).sum())
    seq_diff = int((~np.isclose(a["seq"], b["seq"], atol=1e-7, equal_nan=True)).sum())
    tgt_diff = int((~np.isclose(a["y"], b["y"], atol=1e-7, equal_nan=True)).sum())
    lab_diff = int((a["label"] != b["label"]).sum())
    split_diff = int((a["split"] != b["split"]).sum())
    L = ["# Model inputs: pandas reference table vs Spark table", "",
         f"Rows compared (eval-eligible train/val/test rows used by the models): **{n_rows:,}**; input columns: **{len(names)}** "
         f"(Model B; Model A drops the rain columns).", "",
         "## Differing input columns", "", "| column | rows that differ | share of rows |", "|---|---:|---:|"]
    for c, n in sorted(diff_cols.items()):
        L.append(f"| `{c}` | {n:,} | {100 * n / n_rows:.2f}% |")
    if not diff_cols:
        L.append("| (none) | 0 | 0% |")
    L += ["", f"- Other input columns that differ: **{len([c for c in diff_cols if 'max_to_date' not in c and 'min_to_date' not in c])}**",
          f"- Sequence-tensor cells that differ: **{seq_diff:,}**",
          f"- Target (`anomaly_z_capped`) values that differ: **{tgt_diff:,}**",
          f"- Stress labels that differ: **{lab_diff:,}**; split assignments that differ: **{split_diff:,}**", ""]
    # first 80 differing rows of the extrema columns
    for c in ("gw_max_to_date_m", "gw_min_to_date_m"):
        j = names.index(c)
        bad = np.flatnonzero(~np.isclose(Xa[:, j], Xb[:, j], atol=1e-7, rtol=1e-7, equal_nan=True))
        L += [f"## First 80 differing rows of `{c}` (of {len(bad):,})", "",
              "| well_id | period | pandas | Spark |", "|---|---|---:|---:|"]
        for i in bad[:80]:
            L.append(f"| {a['well'][i]} | {a['period'][i]} | {Xa[i, j]:.4f} | {Xb[i, j]:.4f} |")
        L.append("")
    out = ROOT / "results_spark" / "INPUT_DIFF.md"
    out.write_text("\n".join(L) + "\n")
    (ROOT / "results_spark" / "input_diff.json").write_text(json.dumps({"differing_input_columns": diff_cols, "sequence_cells": seq_diff,
                                                                          "target": tgt_diff, "labels": lab_diff, "splits": split_diff, "rows": n_rows}, indent=1))
    print("\n".join(L[:30]))


if __name__ == "__main__":
    main()
