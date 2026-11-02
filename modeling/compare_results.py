"""Old (pandas table, results/) versus new (Spark table, results_spark/) for every number the paper uses -> results_spark/COMPARISON.md.

    python modeling/compare_results.py [--old results/tables] [--new results_spark/tables]

Nothing is smoothed: every compared number is listed with old, new and difference. Flags:
  NOISE   the change is within seed noise (|diff| <= 2 x the pooled per-seed sd of the two runs; no sd available -> compared with 0.01 m / 0.005)
  MOVED   the change exceeds that noise band
  CHANGES CONCLUSION   a stated conclusion differs (model ranking, TGT-M rank, CI sign/significance, H1/H2 verdicts)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEARNERS = ["SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]
conclusions = []          # (item, old, new, changed)


def md(df, floatfmt=".4f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def noise_flag(old, new, sd_old=None, sd_new=None, floor=0.0):
    d = abs(new - old)
    sds = [s for s in (sd_old, sd_new) if s is not None and not pd.isna(s)]
    band = 2 * np.sqrt(np.mean([s ** 2 for s in sds])) if sds else floor
    band = max(band, floor)
    return "NOISE" if d <= band else "MOVED"


def cmp_table(old, new, keys, cols, sd_suffix=None, floor=0.0):
    m = old.merge(new, on=keys, suffixes=("_old", "_new"))
    rows = []
    for _, r in m.iterrows():
        for c in cols:
            o, n = r[f"{c}_old"], r[f"{c}_new"]
            so = r.get(f"{c.replace('_mean', '')}{sd_suffix}_old") if sd_suffix else None
            sn = r.get(f"{c.replace('_mean', '')}{sd_suffix}_new") if sd_suffix else None
            rows.append({**{k: r[k] for k in keys}, "metric": c, "old": o, "new": n, "diff": n - o,
                         "seed sd (old/new)": "" if so is None or pd.isna(so) else f"{so:.4f} / {sn:.4f}",
                         "flag": noise_flag(o, n, so, sn, floor)})
    return pd.DataFrame(rows)


def rank_of(df, model, col="depth_MAE_mean", variant="B"):
    d = df[(df.variant == variant) & df.model.isin(LEARNERS)].sort_values(col)
    order = list(d.model)
    return order.index(model) + 1, order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", type=Path, default=ROOT / "results" / "tables")
    ap.add_argument("--new", type=Path, default=ROOT / "results_spark" / "tables")
    ap.add_argument("--out", type=Path, default=ROOT / "results_spark" / "COMPARISON.md")
    a = ap.parse_args()
    O = lambda n: pd.read_csv(a.old / n)       # noqa: E731
    N = lambda n: pd.read_csv(a.new / n)       # noqa: E731
    L = ["# Retrain on Spark: old (pandas table) vs new (Spark table)", "",
         f"Old: `{a.old.relative_to(ROOT) if a.old.is_relative_to(ROOT) else a.old}`; new: `{a.new.relative_to(ROOT) if a.new.is_relative_to(ROOT) else a.new}`. "
         "Same protocol, seeds, grids, splits and metrics; only the two columns `gw_max_to_date_m`, `gw_min_to_date_m` (and their normalised "
         "copies `*_rz`) differ in the model inputs (see `INPUT_DIFF.md`).", "",
         "Flags: **NOISE** = |diff| <= 2 x pooled per-seed sd of the two runs (3 seeds; for rules with no sd: 0.01 on MAE-type, 0.005 on the others); "
         "**MOVED** = larger than that. Conclusion changes are collected in the first section.", ""]
    body = []

    # ---------------------------------------------------------------- A. Table III
    so, sn = O("chrono_summary.csv"), N("chrono_summary.csv")
    cols = ["depth_MAE_mean", "depth_RMSE_mean", "depth_R2_mean", "anom_MAE_mean", "anom_R2_mean", "acc_mean", "macro_F1_mean"]
    tA = cmp_table(so, sn, ["model", "variant"], cols, sd_suffix="_std", floor=0.0)
    tA.loc[tA.flag == "NOISE", "flag"] = "NOISE"
    # deterministic rows (no sd): use floor
    nosd = tA["seed sd (old/new)"] == ""
    tA.loc[nosd, "flag"] = [("NOISE" if abs(d) <= (0.01 if "MAE" in m or "RMSE" in m else 0.005) else "MOVED") for d, m in zip(tA.loc[nosd, "diff"], tA.loc[nosd, "metric"])]
    body += ["## A. Main results (Table III): chronological split, test 2018-2022, mean over 3 seeds", "",
             "### A1. Ranking on test depth MAE (7 learners, Model B and Model A)", ""]
    rk = []
    for v in "BA":
        ro, oo = rank_of(so, "TGT-M", variant=v)
        rn, on = rank_of(sn, "TGT-M", variant=v)
        rk.append({"variant": v, "TGT-M rank old": f"{ro} of 7", "TGT-M rank new": f"{rn} of 7", "order old (best to worst)": " < ".join(oo),
                   "order new": " < ".join(on), "same order": oo == on})
        conclusions.append((f"TGT-M rank on depth MAE, Model {v}", f"{ro} of 7", f"{rn} of 7", ro != rn))
        conclusions.append((f"Model ranking by depth MAE, Model {v}", " < ".join(oo), " < ".join(on), oo != on))
    body += [md(pd.DataFrame(rk)), "", "### A2. Every metric", "", md(tA), ""]

    # ---------------------------------------------------------------- B. classification
    co, cn = O("model_metrics_classification.csv"), N("model_metrics_classification.csv")
    tB = cmp_table(co, cn, ["model", "variant", "thresholds"], ["accuracy", "F1_macro", "balanced_accuracy", "MCC", "AUC_ROC_stressed", "AUC_ROC_High"], sd_suffix="_sd")
    nosd = tB["seed sd (old/new)"] == ""
    tB.loc[nosd, "flag"] = ["NOISE" if abs(d) <= 0.005 else "MOVED" for d in tB.loc[nosd, "diff"]]
    body += ["## B. Classification (default thresholds 1/1.5/2 and validation-calibrated)", "", md(tB), ""]
    cc = O("chrono_calibrated.csv"), N("chrono_calibrated.csv")

    # ---------------------------------------------------------------- C. rainfall deltas
    bo, bn = O("bootstrap.csv"), N("bootstrap.csv")
    m = bo.merge(bn, on=["comparison", "metric"], suffixes=("_old", "_new"))
    m["old CI"] = [f"[{l:.4f}, {h:.4f}]" for l, h in zip(m.ci_lo_old, m.ci_hi_old)]
    m["new CI"] = [f"[{l:.4f}, {h:.4f}]" for l, h in zip(m.ci_lo_new, m.ci_hi_new)]
    m["sig old"] = (m.ci_hi_old < 0) | (m.ci_lo_old > 0)
    m["sig new"] = (m.ci_hi_new < 0) | (m.ci_lo_new > 0)
    m["diff of estimate"] = m.diff_new - m.diff_old
    m["flag"] = np.where(m["sig old"] != m["sig new"], "CHANGES CONCLUSION (significance)", np.where(np.sign(m.diff_old) != np.sign(m.diff_new), "sign flip", "same"))
    for _, r in m.iterrows():
        if r["sig old"] != r["sig new"]:
            conclusions.append((f"Significance: {r.comparison} [{r.metric}]", f"CI {r['old CI']}", f"CI {r['new CI']}", True))
    body += ["## C. Rainfall effect (Model B minus Model A) with 95% paired-bootstrap CI over wells", "",
             md(m[["comparison", "metric", "diff_old", "diff_new", "diff of estimate", "old CI", "new CI", "sig old", "sig new", "flag"]]), ""]

    # ---------------------------------------------------------------- D. geographic
    go_, gn = O("geo_summary.csv"), N("geo_summary.csv")
    tD = cmp_table(go_, gn, ["model", "scheme", "variant"], ["depth_MAE", "depth_RMSE", "anom_MAE", "anom_R2", "macro_F1", "AUC_stressed"], floor=0.0)
    tD["flag"] = ["NOISE" if abs(d) <= (0.02 if "MAE" in mm or "RMSE" in mm else 0.01) else "MOVED" for d, mm in zip(tD["diff"], tD.metric)]
    body += ["## D. Unseen-state (geographic) generalisation (single seed per cell, so no sd: bands 0.02 on MAE/RMSE, 0.01 otherwise)", "", md(tD), ""]
    gapo, gapn = O("generalization_gap.csv"), N("generalization_gap.csv")
    g = gapo.merge(gapn, on=["comparison", "what"], suffixes=("_old", "_new"))
    g["sig old"] = (g.ci_hi_old < 0) | (g.ci_lo_old > 0)
    g["sig new"] = (g.ci_hi_new < 0) | (g.ci_lo_new > 0)
    for _, r in g.iterrows():
        if r["sig old"] != r["sig new"]:
            conclusions.append((f"Seen-to-unseen gap significance: {r.comparison} / {r.what}", f"CI [{r.ci_lo_old:.4f}, {r.ci_hi_old:.4f}]", f"CI [{r.ci_lo_new:.4f}, {r.ci_hi_new:.4f}]", True))
    body += ["### D2. Post-hoc seen-to-unseen gap (95% CI)", "", md(g[["comparison", "what", "estimate_old", "estimate_new", "ci_lo_old", "ci_hi_old", "ci_lo_new", "ci_hi_new", "sig old", "sig new"]]), ""]

    # ---------------------------------------------------------------- E. H1
    io, in_ = O("interval_summary.csv"), N("interval_summary.csv")
    tE = io.merge(in_, on="method", suffixes=("_old", "_new"))
    tE["flag"] = ["NOISE" if abs(a_ - b_) <= 2 * np.sqrt((sa ** 2 + sb ** 2) / 2) else "MOVED" for a_, b_, sa, sb in
                  zip(tE.interval_score_old, tE.interval_score_new, tE.interval_score_sd_old, tE.interval_score_sd_new)]
    body += ["## E. H1: 80% interval quality (interval score, lower is better; coverage target 80% +/- 3)", "",
             md(tE[["method", "coverage_old", "coverage_new", "width_old", "width_new", "interval_score_old", "interval_score_new", "interval_score_sd_old", "interval_score_sd_new", "flag"]]), ""]
    ico, icn = O("interval_comparison.csv"), N("interval_comparison.csv")
    ic = ico.merge(icn, on="comparison", suffixes=("_old", "_new"))
    ic["sig old"] = ic.ci_hi_old < 0
    ic["sig new"] = ic.ci_hi_new < 0
    ic["flag"] = np.where(ic["sig old"] != ic["sig new"], "CHANGES CONCLUSION", "same")
    body += ["### E2. TGT-M minus competitor, interval score (negative = TGT-M better), 95% CI", "",
             md(ic[["comparison", "diff_interval_score_old", "diff_interval_score_new", "ci_lo_old", "ci_hi_old", "ci_lo_new", "ci_hi_new", "sig old", "sig new", "flag"]], ".4f"), ""]

    def h1(io_, ico_):
        nat = ico_[ico_.comparison == "TGT-M native minus XGBoost-QR native"].iloc[0]
        cq = ico_[ico_.comparison == "TGT-M + CQR minus XGBoost-QR + CQR"].iloc[0]
        cov = float(io_[io_.method == "TGT-M native"].coverage.iloc[0])
        ok = (nat.ci_hi < 0) and (cq.ci_hi < 0) and abs(cov - 0.80) <= 0.03
        detail = (f"native vs native {nat.diff_interval_score:+.4f} [{nat.ci_lo:.4f}, {nat.ci_hi:.4f}], CQR vs CQR {cq.diff_interval_score:+.4f} "
                  f"[{cq.ci_lo:.4f}, {cq.ci_hi:.4f}], native coverage {100 * cov:.1f}%")
        part = (nat.ci_hi < 0) != (cq.ci_hi < 0)
        return ("SUPPORTED" if ok else ("PARTIALLY supported" if part else "NOT supported")), detail

    h1o, h1n = h1(io, ico), h1(in_, icn)
    conclusions.append(("H1 (strict pre-registered rule)", f"{h1o[0]} ({h1o[1]})", f"{h1n[0]} ({h1n[1]})", h1o[0] != h1n[0]))

    # ---------------------------------------------------------------- F. H2
    ro, rn = O("robust_degradation.csv"), N("robust_degradation.csv")
    tF = ro.merge(rn, on=["comparison", "what"], suffixes=("_old", "_new"))
    tF["sig old"] = (tF.ci_hi_old < 0) | (tF.ci_lo_old > 0)
    tF["sig new"] = (tF.ci_hi_new < 0) | (tF.ci_lo_new > 0)
    tF["flag"] = np.where(tF["sig old"] != tF["sig new"], "CHANGES CONCLUSION (significance)", "same")
    body += ["## F. H2: robustness to missing recent history", "", "### F1. Degradation D = anomaly MAE(p=0.6) - MAE(p=0) and differences, 95% CI", "",
             md(tF[["comparison", "what", "estimate_old", "estimate_new", "ci_lo_old", "ci_hi_old", "ci_lo_new", "ci_hi_new", "sig old", "sig new", "flag"]]), ""]
    for _, r in tF[tF["sig old"] != tF["sig new"]].iterrows():
        conclusions.append((f"H2 CI significance: {r.comparison} / {r.what}", f"[{r.ci_lo_old:.4f}, {r.ci_hi_old:.4f}]", f"[{r.ci_lo_new:.4f}, {r.ci_hi_new:.4f}]", True))
    so_, sn_ = O("robust_summary.csv"), N("robust_summary.csv")
    ms = so_.merge(sn_, on=["model", "p"], suffixes=("_old", "_new"))
    ms["diff"] = ms.mean_new - ms.mean_old
    ms["flag"] = ["NOISE" if abs(d) <= 2 * np.sqrt((a_ ** 2 + b_ ** 2) / 2) else "MOVED" for d, a_, b_ in zip(ms["diff"], ms.std_old, ms.std_new)]
    body += ["### F2. Anomaly MAE by masking share p (mean of 3 seeds)", "", md(ms[["model", "p", "mean_old", "mean_new", "diff", "std_old", "std_new", "flag"]]), ""]

    def h2(rd, rs):
        def diffrow(other):
            r = rd[(rd.comparison == f"TGT-M minus {other}") & (rd.what == "difference of degradations")].iloc[0]
            return r
        a_ok = all(diffrow(o).ci_hi < 0 for o in ("LSTM", "CNN1D"))
        a_txt = "; ".join(f"vs {o}: {diffrow(o).estimate:+.4f} [{diffrow(o).ci_lo:.4f}, {diffrow(o).ci_hi:.4f}]" for o in ("LSTM", "CNN1D"))
        p6 = rs[np.isclose(rs.p, 0.6)].set_index("model")["mean"]
        best = p6[p6.index.isin(LEARNERS)].idxmin()
        return a_ok, a_txt, best == "TGT-M", f"lowest at p=0.6: {best} ({p6[best]:.4f}); TGT-M {p6['TGT-M']:.4f}"

    ho, hn = h2(ro, so_), h2(rn, sn_)
    conclusions.append(("H2a (D of TGT-M significantly smaller than LSTM and CNN1D)", f"{'SUPPORTED' if ho[0] else 'NOT supported'} ({ho[1]})", f"{'SUPPORTED' if hn[0] else 'NOT supported'} ({hn[1]})", ho[0] != hn[0]))
    conclusions.append(("H2b (TGT-M lowest anomaly MAE at p=0.6)", f"{'SUPPORTED' if ho[2] else 'NOT supported'} ({ho[3]})", f"{'SUPPORTED' if hn[2] else 'NOT supported'} ({hn[3]})", ho[2] != hn[2]))

    # ---------------------------------------------------------------- G. label sensitivity counts (from the data tables)
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    from gwstress import stress
    cols_ = ["well_id", "date", "year", "stress_category", "anomaly_z", "n_prior_same_season"]
    po = pd.read_parquet(ROOT / "processed" / "gw_features_by_state")
    ps = pd.read_parquet(ROOT / "processed" / "spark" / "gw_features_by_state")
    cnt = pd.DataFrame({"pandas table": po.stress_category.astype(str).value_counts(), "Spark table": ps.stress_category.astype(str).value_counts()})
    cnt["diff"] = cnt["Spark table"] - cnt["pandas table"]
    body += ["## G. Label counts and threshold sensitivity (computed from the two preprocessed tables)", "", "### G1. Stress category counts", "", cnt.reset_index(names="category").to_markdown(index=False), ""]
    try:
        co_, ch_o = stress.sensitivity(po)
        cs_, ch_s = stress.sensitivity(ps)
        sens = ch_o.merge(ch_s, on="threshold set", suffixes=(" (pandas)", " (Spark)"))
        body += ["### G2. Readings that change category under other threshold sets", "", md(sens, ".2f"), ""]
        conclusions.append(("Label sensitivity counts (threshold sets)", "see G2", "see G2", not co_.equals(cs_)))
    except Exception as e:          # keep the report honest if the helper's signature differs
        body += [f"(G2 not computed: {e})", ""]
    conclusions.append(("Stress label counts", cnt["pandas table"].to_dict().__repr__(), cnt["Spark table"].to_dict().__repr__(), bool((cnt["diff"] != 0).any())))

    # ---------------------------------------------------------------- H. confusion matrices
    cmo = json.loads((a.old / "confusion_matrices.json").read_text())
    cmn = json.loads((a.new / "confusion_matrices.json").read_text())
    cls = ["Normal", "Watch", "Moderate Stress", "High Stress"]
    body += ["## H. TGT-M confusion matrices (rows = true class, columns = predicted; default thresholds, mean over seeds as saved)", ""]
    for key in ("TGT-M|A", "TGT-M|B"):
        for tag, d in (("old", cmo), ("new", cmn)):
            if key in d:
                body += [f"**{key}, {tag}**", "", pd.DataFrame(d[key], index=[f"true {c}" for c in cls], columns=[f"pred {c}" for c in cls]).round(1).to_markdown(), ""]
        if key in cmo and key in cmn:
            dd = np.array(cmn[key], float) - np.array(cmo[key], float)
            body += [f"**{key}, new - old**", "", pd.DataFrame(dd, index=[f"true {c}" for c in cls], columns=[f"pred {c}" for c in cls]).round(1).to_markdown(), ""]

    # ---------------------------------------------------------------- summary on top
    flagged = [c for c in conclusions if c[3]]
    top = ["## 0. Stated conclusions: old vs new", "", "| item | old | new | changed? |", "|---|---|---|---|"]
    for item, o, n, ch in conclusions:
        top.append(f"| {item} | {o} | {n} | {'**CHANGES CONCLUSION**' if ch else 'no'} |")
    top += ["", f"**{len(flagged)} of {len(conclusions)} checked conclusions changed.**" + ("" if flagged else " None of the model ranking, TGT-M rank, CI significance flags or H1/H2 verdicts flipped."), ""]
    movedA = tA[tA.flag == "MOVED"]
    top += [f"Table III: {len(tA)} numbers compared; {len(movedA)} moved by more than seed noise" + (":" if len(movedA) else "."), ""]
    if len(movedA):
        top += [md(movedA[["model", "variant", "metric", "old", "new", "diff", "seed sd (old/new)"]]), ""]
    a.out.write_text("\n".join(L + top + body) + "\n")
    tA.to_csv(a.out.with_name("comparison_table3.csv"), index=False)
    print("\n".join(top))


if __name__ == "__main__":
    main()
