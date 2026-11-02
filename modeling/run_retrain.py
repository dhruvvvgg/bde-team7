"""Resumable full retraining: the same protocol and steps as run_pipeline.py --mode full, with a completion check before every job.

A job whose output already exists and is complete is skipped, so the run can be interrupted (container restart, Ctrl-C) and restarted
with the same command; at most the jobs that were running at that moment are repeated. Nothing about the protocol differs from
run_pipeline.py: same scripts, same arguments, same seeds, grids, splits and metrics. Only the two environment variables of paths.py
decide where inputs come from and where results go:

    BDE_PROCESSED=processed/spark BDE_RESULTS=results_spark python modeling/run_retrain.py [--steps chrono,geo,robust,evaluate]

Parallelism: single-thread jobs are run NCPU at a time (run_pipeline.py used a smaller pool); thread counts only change floating-point
summation order, not the protocol.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_pipeline as RP  # noqa: E402
from paths import PROCESSED, RESULTS  # noqa: E402

NCPU = RP.NCPU
CH = RESULTS / "chrono"
JOBS = CH / "jobs"
GEO = RESULTS / "geo"
ROB = RESULTS / "robust"
RP.LOGS.mkdir(parents=True, exist_ok=True)


def complete_json(path, variants=("A", "B")):
    try:
        j = json.loads(Path(path).read_text())
    except Exception:
        return False
    return all(v in j.get("variants", {}) and len(j["variants"][v].get("seeds", [1, 2, 3])) >= 1 for v in variants)


def todo(jobs, done):
    keep = [j for j in jobs if not done(j)]
    print(f"   {len(jobs) - len(keep)} of {len(jobs)} already complete, {len(keep)} to run", flush=True)
    return keep


def pool(jobs, workers):
    if jobs:
        RP.pool(jobs, workers)


def step_chrono():
    models = ["XGBoost", "RF", "SVR", "ANN", "CNN1D", "LSTM"]
    jobs = [(["run_chrono.py", m], {"BDE_THREADS": 1}, RP.LOGS / f"chrono_{m}.log") for m in models]
    print(f"[chrono] {len(jobs)} learners, 1 thread each", flush=True)
    pool(todo(jobs, lambda j: complete_json(CH / f"{j[0][1]}.json")), NCPU)
    tune = [(["tgtm_job.py", v, str(c), "0"], {"BDE_THREADS": 1}, RP.LOGS / f"tgtm_{v}_c{c}_s0.log") for v in "AB" for c in range(3)]
    print("[chrono] TGT-M tuning (3 configs x 2 variants, seed 0)", flush=True)
    pool(todo(tune, lambda j: (JOBS / f"TGT-M_{j[0][1]}_c{j[0][2]}_s0.json").exists()), NCPU)
    best = {}
    for v in "AB":
        t = [json.loads((JOBS / f"TGT-M_{v}_c{c}_s0.json").read_text())["val_anom_MAE"] for c in range(3)]
        best[v] = t.index(min(t))
    seeds = [(["tgtm_job.py", v, str(best[v]), str(s)], {"BDE_THREADS": 1}, RP.LOGS / f"tgtm_{v}_c{best[v]}_s{s}.log") for v in "AB" for s in (1, 2)]
    print(f"[chrono] TGT-M refit, best configs {best}", flush=True)
    pool(todo(seeds, lambda j: (JOBS / f"TGT-M_{j[0][1]}_c{j[0][2]}_s{j[0][3]}.json").exists()), NCPU)
    RP.run(["tgtm_merge.py"])


def step_geo():
    jobs = [(["run_geo.py", m, "g5"], {"BDE_THREADS": 1}, RP.LOGS / f"geo_{m}_g5.log") for m in RP.TABULAR + RP.DEEP]
    jobs += [(["run_geo.py", m, "loso"], {"BDE_THREADS": 1}, RP.LOGS / f"geo_{m}_loso.log") for m in ("XGBoost", "SVR")]
    print(f"[geo] {len(jobs)} jobs", flush=True)
    pool(todo(jobs, lambda j: complete_json(GEO / f"{j[0][1]}_{j[0][2]}.json")), NCPU)
    folds = [(["run_geo.py", "TGT-M", "g5", v, str(g)], {"BDE_THREADS": 1}, RP.LOGS / f"geo_TGT-M_{v}_{g}.log") for v in "AB" for g in range(5)]
    print("[geo] TGT-M grouped 5-fold fold jobs", flush=True)
    pool(todo(folds, lambda j: (GEO / f"part_TGT-M_g5_{j[0][3]}_{j[0][4]}.npy").exists()), NCPU)
    if not complete_json(GEO / "TGT-M_g5.json"):
        RP.run(["run_geo.py", "TGT-M", "g5", "merge"], {"BDE_THREADS": 1}, RP.LOGS / "geo_TGT-M_merge.log")


def step_robust():
    ms = ["TGT-M", "LSTM", "CNN1D", "RF", "ANN", "XGBoost", "SVR"]
    jobs = [(["robust_job.py", m, str(s)], {"BDE_THREADS": 1}, RP.LOGS / f"robust_{m}_{s}.log") for m in ms for s in (0, 1, 2)]
    print(f"[robust] {len(jobs)} retrain+mask jobs", flush=True)
    pool(todo(jobs, lambda j: (ROB / f"pred_{j[0][1]}_s{j[0][2]}_p60.npy").exists()), NCPU)


def step_evaluate():
    for s in ("aggregate.py", "metrics_full.py", "robust_eval.py", "interval_eval.py", "gap_test.py", "export_powerbi.py"):
        rc = RP.run([s], {"BDE_THREADS": NCPU}, RP.LOGS / f"eval_{s[:-3]}.log")
        if rc:
            raise SystemExit(f"{s} failed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="chrono,geo,robust,evaluate")
    a = ap.parse_args()
    print(f"input table: {PROCESSED}\nresults:     {RESULTS}", flush=True)
    t0 = time.time()
    for s in a.steps.split(","):
        print(f"=== {s} === ({time.time() - t0:.0f}s elapsed)", flush=True)
        {"chrono": step_chrono, "geo": step_geo, "robust": step_robust, "evaluate": step_evaluate}[s]()
    print(f"retrain finished in {time.time() - t0:.0f}s", flush=True)
