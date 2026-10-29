"""Run the whole modeling study from the committed processed/ tables (CPU only, works on Linux/Colab/Kaggle/Windows).

usage:
  python modeling/run_pipeline.py --mode quick            # XGBoost + ANN, 1 seed, ~10-20 min, writes to results_quick/ (does not touch results/)
  python modeling/run_pipeline.py --mode full             # everything, several hours; overwrites results/
  python modeling/run_pipeline.py --mode full --steps chrono,geo   # only some steps
steps (in order): chrono, geo, robust, evaluate, dashboard
"""
import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOD = ROOT / "modeling"
LOGS = ROOT / "results" / "logs"
NCPU = os.cpu_count() or 4
TABULAR = ["XGBoost", "RF", "SVR"]
DEEP = ["ANN", "CNN1D", "LSTM"]
ALL = TABULAR + DEEP + ["TGT-M"]


def run(cmd, env_extra=None, log=None):
    env = dict(os.environ)
    env.update({k: str(v) for k, v in (env_extra or {}).items()})
    t0 = time.time()
    out = open(log, "w") if log else None
    r = subprocess.run([sys.executable, "-u"] + cmd, cwd=MOD, env=env, stdout=out or None, stderr=subprocess.STDOUT if out else None)
    if out:
        out.close()
    status = "ok" if r.returncode == 0 else f"FAILED ({r.returncode}), see {log}"
    print(f"  {' '.join(cmd)}  [{time.time() - t0:.0f}s] {status}", flush=True)
    return r.returncode


def pool(jobs, workers):
    """jobs: list of (cmd, env, log). Runs them in parallel and stops the pipeline if one fails."""
    with ThreadPoolExecutor(workers) as ex:
        codes = list(ex.map(lambda j: run(*j), jobs))
    if any(codes):
        raise SystemExit("a job failed; fix it and rerun (finished jobs are not repeated only if you pass --steps)")


def step_chrono():
    LOGS.mkdir(parents=True, exist_ok=True)
    th = max(1, NCPU // 4)
    jobs = [(["run_chrono.py", m], {"BDE_THREADS": th}, LOGS / f"chrono_{m}.log") for m in TABULAR + DEEP]
    print(f"[chrono] {len(jobs)} models in parallel ({th} threads each), then TGT-M", flush=True)
    pool(jobs, 3)
    print("[chrono] TGT-M: tune 3 configs x 2 variants (parallel, 1 thread each), then seeds 1 and 2 for the best config", flush=True)
    pool([(["tgtm_job.py", v, str(c), "0"], {"BDE_THREADS": 1}, LOGS / f"tgtm_{v}_c{c}_s0.log") for v in "AB" for c in range(3)], max(1, min(6, NCPU)))
    import json
    best = {}
    for v in "AB":
        t = [json.loads((ROOT / "results" / "chrono" / "jobs" / f"TGT-M_{v}_c{c}_s0.json").read_text())["val_anom_MAE"] for c in range(3)]
        best[v] = t.index(min(t))
    pool([(["tgtm_job.py", v, str(best[v]), str(s)], {"BDE_THREADS": 1}, LOGS / f"tgtm_{v}_c{best[v]}_s{s}.log") for v in "AB" for s in (1, 2)], 4)
    run(["tgtm_merge.py"])


def step_geo():
    th = max(1, NCPU // 4)
    jobs = [(["run_geo.py", m, "g5"], {"BDE_THREADS": th}, LOGS / f"geo_{m}_g5.log") for m in TABULAR + DEEP]
    jobs += [(["run_geo.py", m, "loso"], {"BDE_THREADS": th}, LOGS / f"geo_{m}_loso.log") for m in ("XGBoost", "SVR")]
    print(f"[geo] {len(jobs)} jobs", flush=True)
    pool(jobs, 3)
    print("[geo] TGT-M grouped 5-fold as parallel fold jobs, then merge", flush=True)
    pool([(["run_geo.py", "TGT-M", "g5", v, str(g)], {"BDE_THREADS": 1}, LOGS / f"geo_TGT-M_{v}_{g}.log") for v in "AB" for g in range(5)], max(1, min(10, NCPU)))
    run(["run_geo.py", "TGT-M", "g5", "merge"])


def step_robust():
    print("[robust] retrain Model B (3 seeds) and predict under history masking", flush=True)
    jobs = [(["robust_job.py", m, str(s)], {"BDE_THREADS": 1}, LOGS / f"robust_{m}_{s}.log") for m in ["TGT-M", "LSTM", "CNN1D", "RF", "ANN", "XGBoost", "SVR"] for s in (0, 1, 2)]
    pool(jobs, max(1, min(11, NCPU)))


def step_evaluate():
    for s in ("aggregate.py", "metrics_full.py", "robust_eval.py", "interval_eval.py", "gap_test.py", "export_powerbi.py"):
        run([s], {"BDE_THREADS": NCPU})


def step_dashboard():
    run(["make_dashboard.py"])


def quick():
    out = ROOT / "results_quick"
    print(f"[quick] XGBoost + ANN, seed 0 only, output in {out.name}/ (results/ is untouched)", flush=True)
    env = {"BDE_SEEDS": "0", "BDE_RESULTS": out, "BDE_THREADS": max(1, NCPU // 2)}
    pool([(["run_chrono.py", m], env, ROOT / f"results_quick_{m}.log") for m in ("XGBoost", "ANN")], 2)
    import json
    print("\nmodel  variant  test depth MAE  anomaly R2")
    for m in ("XGBoost", "ANN"):
        j = json.loads((out / "chrono" / f"{m}.json").read_text())
        for v, r in j["variants"].items():
            t = r["seeds"][0]["test"]
            print(f"{m:8s} {v}   {t['depth_MAE']:.3f}          {t['anom_R2']:.3f}")
    print("\nExpected: XGBoost A 1.264 / B 1.120; ANN seed 0 A 1.300 / B 1.142 (the report shows 3-seed means: ANN 1.270 / 1.125). Differences of about 0.01 m are normal.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["quick", "full"], default="quick")
    ap.add_argument("--steps", default="chrono,geo,robust,evaluate,dashboard")
    a = ap.parse_args()
    if a.mode == "quick":
        quick()
    else:
        for s in a.steps.split(","):
            print(f"=== {s} ===", flush=True)
            {"chrono": step_chrono, "geo": step_geo, "robust": step_robust, "evaluate": step_evaluate, "dashboard": step_dashboard}[s]()
        print("pipeline finished")
