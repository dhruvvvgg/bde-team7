"""Pick the best config per variant (validation anomaly MAE, seed 0) and write TGT-M__<v>.json + pred files in run_chrono format."""
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parents[1] / "results" / "chrono"
jobs = root / "jobs"
for v in ("A", "B"):
    tune = [json.loads((jobs / f"TGT-M_{v}_c{i}_s0.json").read_text()) for i in range(3)]
    best = min(range(3), key=lambda i: tune[i]["val_anom_MAE"])
    seeds = []
    for s in (0, 1, 2):
        r = json.loads((jobs / f"TGT-M_{v}_c{best}_s{s}.json").read_text())
        shutil.copy(jobs / f"pred_TGT-M_{v}_c{best}_s{s}.npy", root / f"pred_TGT-M_{v}_s{s}.npy")
        seeds.append({"seed": s, "val": r["val"], "test": r["test"], "PI80_coverage": r["PI80_coverage"], "PI80_width": r["PI80_width"],
                      "vsn_rain_weight": r["vsn_rain_weight"], "vsn_w_mean": r["vsn_w_mean"],
                      "attn_mean_by_step": r["attn_mean_by_step"], "epochs": r["epochs"]})
    rec = {"tuning": [{"cfg": t["cfg"], "val_anom_MAE": t["val_anom_MAE"]} for t in tune], "seeds": seeds,
           "best_cfg": tune[best]["cfg"], "time_s": sum(t["time_s"] for t in tune)}
    (root / f"TGT-M__{v}.json").write_text(json.dumps({"model": "TGT-M", "variants": {v: rec}}, indent=1))
    print(v, "best cfg", best, [round(t["val_anom_MAE"], 4) for t in tune])
