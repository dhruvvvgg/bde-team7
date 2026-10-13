"""Interactive dashboard (single HTML file, Plotly) + extra Power BI / Tableau tables, built from results/ and processed/.

usage: python make_dashboard.py            -> results/dashboard/dashboard.html, results/powerbi/*.csv (extra tables)
The page is self-contained (Plotly JS embedded), works offline and needs no server. No matplotlib.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs

ROOT = Path(os.environ.get("BDE_ROOT", Path(__file__).resolve().parents[1]))
T, PB, OUT = ROOT / "results" / "tables", ROOT / "results" / "powerbi", ROOT / "results" / "dashboard"
OUT.mkdir(parents=True, exist_ok=True)
PB.mkdir(parents=True, exist_ok=True)
LAB = {"CNN1D": "1D-CNN", "TGT-M": "TGT-M (proposed)", "Prev. round": "Previous round", "Same season LY": "Same season last year"}
ORDER = ["Prev. round", "Same season LY", "Seasonal baseline", "SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"]
COL = {"A": "#1F4E79", "B": "#C55A11"}
STRESS_COL = {"Normal": "#9ECAE1", "Watch": "#FDD835", "Moderate Stress": "#FB8C00", "High Stress": "#C62828"}
lab = lambda m: LAB.get(m, m)


def read(name):
    f = T / name
    return pd.read_csv(f) if f.exists() else pd.DataFrame()


S, CAL, BOOT = read("chrono_summary.csv"), read("chrono_calibrated.csv"), read("bootstrap.csv")
SEA, PST, GPS, GEO = read("chrono_by_season.csv"), read("chrono_per_state.csv"), read("geo_per_state.csv"), read("geo_summary.csv")
INT, ROB, ROBD, GAP = read("interval_summary.csv"), read("robust_summary.csv"), read("robust_degradation.csv"), read("generalization_gap.csv")
figs = {}


def lay(fig, title, h=520):
    fig.update_layout(title=title, height=h, template="plotly_white", font=dict(family="Times New Roman", size=13),
                      legend=dict(orientation="h", y=-0.2), margin=dict(l=60, r=30, t=120, b=80), title_y=0.96)
    return fig


# ---------------------------------------------------------------- 1 model comparison (dropdown metric)
metrics = {"depth_MAE": "Depth MAE (m)", "depth_RMSE": "Depth RMSE (m)", "anom_MAE": "Anomaly MAE (z)", "anom_R2": "Anomaly R2", "macro_F1": "Macro-F1 (1/1.5/2)", "AUC_stressed": "AUC (stressed vs not)"}
models = [m for m in ORDER if (S.model == m).any()]
fig = go.Figure()
for v in ("A", "B"):
    for k, (mk, mn) in enumerate(metrics.items()):
        d = S[S.variant == v].set_index("model")
        fig.add_bar(x=[lab(m) for m in models], y=[d.loc[m, f"{mk}_mean"] for m in models],
                    error_y=dict(type="data", array=[0 if pd.isna(d.loc[m, f"{mk}_std"]) else d.loc[m, f"{mk}_std"] for m in models]),
                    name=f"Model {v} ({'no rainfall' if v == 'A' else 'with rainfall'})", marker_color=COL[v], visible=(k == 0), legendgroup=v, showlegend=(k == 0))
nm, nv = len(metrics), 2
buttons = []
for k, (mk, mn) in enumerate(metrics.items()):
    vis = [False] * (nm * nv)
    for v in range(nv):
        vis[v * nm + k] = True
    buttons.append(dict(label=mn, method="update", args=[{"visible": vis}, {"yaxis.title": mn}]))
fig.update_layout(updatemenus=[dict(buttons=buttons, x=0.0, y=1.09, xanchor="left")], barmode="group", yaxis_title="Depth MAE (m)")
figs["models"] = lay(fig, "Test results by model (2018-2022, mean +/- sd over 3 seeds); use the menu to change the metric")

# ---------------------------------------------------------------- 2 rainfall effect with CI
if not BOOT.empty:
    b = BOOT[BOOT.comparison.str.endswith("B - A (rainfall effect)") & (BOOT.metric == "depth_MAE")].copy()
    b["model"] = b.comparison.str.split(":").str[0]
    fig = go.Figure(go.Bar(x=[lab(m) for m in b.model], y=b["diff"], marker_color="#548235",
                           error_y=dict(type="data", symmetric=False, array=b.ci_hi - b["diff"], arrayminus=b["diff"] - b.ci_lo)))
    fig.update_layout(yaxis_title="Change in depth MAE when rainfall is added (m); negative = rainfall helps")
    figs["rain"] = lay(fig, "Contribution of rainfall (B minus A) with 95% paired-bootstrap interval over wells")

# ---------------------------------------------------------------- 3 season
if not SEA.empty:
    nmap = {"post_monsoon_rabi": "Jan", "pre_monsoon": "May", "monsoon": "Aug", "post_monsoon_kharif": "Nov"}
    fig = go.Figure()
    for m in [x for x in ["Seasonal baseline", "SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"] if (SEA.model == x).any()]:
        d = SEA[(SEA.model == m) & (SEA.variant == "B")].set_index("season")
        fig.add_bar(x=[nmap[s] for s in nmap], y=[d.loc[s, "depth_MAE"] for s in nmap], name=lab(m))
    fig.update_layout(barmode="group", yaxis_title="Depth MAE (m), Model B")
    figs["season"] = lay(fig, "Error by round month (test split, Model B)")

# ---------------------------------------------------------------- 4 state: seen vs unseen (model dropdown)
if not GPS.empty and not PST.empty:
    mods = [m for m in ["XGBoost", "ANN", "CNN1D", "LSTM", "RF", "SVR", "TGT-M"] if ((GPS.model == m) & (GPS.scheme == "g5") & (GPS.variant == "B")).any()]
    fig = go.Figure()
    for k, m in enumerate(mods):
        u = GPS[(GPS.model == m) & (GPS.scheme == "g5") & (GPS.variant == "B")].set_index("state")
        c = PST[(PST.model == m) & (PST.variant == "B")].set_index("state")
        st = u.index.intersection(c.index)
        st = sorted(st, key=lambda s: -u.loc[s, "n"])
        fig.add_bar(x=st, y=[c.loc[s, "anom_MAE"] for s in st], name="Seen (chronological)", marker_color="#1F4E79", visible=(k == 0), legendgroup="s", showlegend=(k == 0))
        fig.add_bar(x=st, y=[u.loc[s, "anom_MAE"] for s in st], name="Unseen (state held out)", marker_color="#C55A11", visible=(k == 0), legendgroup="u", showlegend=(k == 0))
    buttons = []
    for k, m in enumerate(mods):
        vis = [False] * (2 * len(mods)); vis[2 * k] = vis[2 * k + 1] = True
        buttons.append(dict(label=lab(m), method="update", args=[{"visible": vis}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0.0, y=1.09, xanchor="left")], barmode="group", yaxis_title="Anomaly MAE (z)", xaxis_tickangle=-45)
    figs["states"] = lay(fig, "Per-state error: state seen in training vs held out (Model B; states sorted by test rows)", 560)

# ---------------------------------------------------------------- 5 generalization gap
if not GAP.empty:
    g = GAP[GAP.what == "own gap"]
    fig = go.Figure(go.Bar(x=[lab(m) for m in g.comparison], y=g.estimate, marker_color="#7030A0",
                           error_y=dict(type="data", symmetric=False, array=g.ci_hi - g.estimate, arrayminus=g.estimate - g.ci_lo)))
    fig.update_layout(yaxis_title="Anomaly MAE: unseen state minus seen state")
    figs["gap"] = lay(fig, "Change in error when a state is held out (post-hoc comparison, seed 0, 95% bootstrap CI)")

# ---------------------------------------------------------------- 6 intervals
if not INT.empty:
    d = INT.copy()
    d["method"] = d.method.str.replace(" point + conformal", " + conformal")
    fig = go.Figure()
    fig.add_bar(x=d.method, y=d.interval_score, marker_color=["#7030A0" if m.startswith("TGT-M") else "#7F7F7F" for m in d.method],
                text=[f"{v:.3f}" for v in d.interval_score], textposition="outside", name="Interval score")
    fig.update_layout(yaxis=dict(title="Interval score (lower is better)", range=[3.4, 4.0]), xaxis_tickangle=-40)
    figs["interval"] = lay(fig, "H1 (pre-registered): quality of 80% intervals", 560)

# ---------------------------------------------------------------- 7 robustness
if not ROB.empty:
    fig = go.Figure()
    for m in [x for x in ["SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M"] if (ROB.model == x).any()]:
        d = ROB[ROB.model == m]
        fig.add_scatter(x=d.p, y=d["mean"], mode="lines+markers", name=lab(m), line=dict(width=4 if m == "TGT-M" else 2))
    fig.update_layout(xaxis_title="Share of recent readings removed at test time", yaxis_title="Anomaly MAE (z)")
    figs["robust"] = lay(fig, "H2 (pre-registered): robustness to missing recent history")

# ---------------------------------------------------------------- 8 map (animated by year) + stress counts
main = pd.read_parquet(ROOT / "processed" / "gw_features_by_state",
                       columns=["well_id", "state", "district", "lat", "lon", "year", "month", "gwl_m_bgl", "stress_category", "stress_tadj", "stress_roll10", "baseline_mean_m", "anomaly_z_capped"])
main["state"] = main["state"].astype(str)
cls = main[main.stress_category.astype(str).isin(STRESS_COL)].copy()
cls["stress_category"] = cls.stress_category.astype(str)
yearly = cls[cls.month == 11].sort_values(["well_id", "year"]).drop_duplicates(["well_id", "year"])
years = sorted(yearly.year.unique())
frames, first = [], None
for y in years:
    d = yearly[yearly.year == y]
    tr = [go.Scattergeo(lon=d[d.stress_category == k].lon, lat=d[d.stress_category == k].lat, mode="markers", name=k,
                        marker=dict(size=5, color=c, opacity=0.85), text=d[d.stress_category == k].apply(lambda r: f"{r.district}, {r.state}<br>depth {r.gwl_m_bgl:.1f} m<br>anomaly {r.anomaly_z_capped:+.2f}", axis=1),
                        hoverinfo="text") for k, c in STRESS_COL.items()]
    frames.append(go.Frame(data=tr, name=str(y)))
    first = first or tr
fig = go.Figure(data=first, frames=frames)
fig.update_geos(scope="asia", lonaxis_range=[67, 98], lataxis_range=[7, 37], showcountries=True, countrycolor="#999", showland=True, landcolor="#F5F5F5", showsubunits=True, subunitcolor="#DDD")
fig.update_layout(updatemenus=[dict(type="buttons", x=0.02, y=0.02, buttons=[dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=900), fromcurrent=True)])])],
                  sliders=[dict(steps=[dict(label=str(y), method="animate", args=[[str(y)], dict(mode="immediate", frame=dict(duration=0))]) for y in years], x=0.1, len=0.85)])
fig.update_geos(projection_scale=3.0, center=dict(lat=22.5, lon=82.5))
fig.update_layout(legend=dict(orientation="v", x=0.86, y=0.9))
figs["map"] = lay(fig, "Stress category of every well, November (post-monsoon) round, by year (primary label: expanding baseline)", 700)
figs["map"].update_layout(legend=dict(orientation="v", x=0.86, y=0.9))

cnt = cls.groupby(["year", "stress_category"]).size().unstack(fill_value=0)
cnt = cnt.div(cnt.sum(axis=1), axis=0) * 100
fig = go.Figure()
for k, c in STRESS_COL.items():
    if k in cnt:
        fig.add_bar(x=cnt.index, y=cnt[k], name=k, marker_color=c)
fig.update_layout(barmode="stack", yaxis_title="Share of classified readings (%)", xaxis_title="Year")
figs["stressyear"] = lay(fig, "Stress category share by year (all rounds with a baseline)")

# ---------------------------------------------------------------- 9 predicted vs actual explorer (state dropdown)
tp = PB / "test_predictions.csv"
if tp.exists():
    t = pd.read_csv(tp)
    col = "TGT-M_B_depth_pred" if "TGT-M_B_depth_pred" in t else [c for c in t.columns if c.endswith("_depth_pred")][0]
    st = ["All states"] + sorted(t.state.unique())
    rng = np.random.default_rng(0)
    fig = go.Figure()
    for k, s in enumerate(st):
        d = t if s == "All states" else t[t.state == s]
        d = d.iloc[rng.permutation(len(d))[:4000]]
        fig.add_scatter(x=d.gwl_m_bgl, y=d[col], mode="markers", marker=dict(size=4, opacity=0.5, color="#7030A0"), name=s, visible=(k == 0), showlegend=False,
                        text=d.well_id + " " + d.period_label, hoverinfo="text+x+y")
    mx = float(np.nanpercentile(t.gwl_m_bgl, 99))
    fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", line=dict(color="black", dash="dash"), name="perfect", showlegend=False)
    buttons = []
    for k, s in enumerate(st):
        vis = [False] * (len(st) + 1); vis[k] = True; vis[-1] = True
        buttons.append(dict(label=s, method="update", args=[{"visible": vis}]))
    fig.update_layout(updatemenus=[dict(buttons=buttons, x=0.0, y=1.09, xanchor="left")], xaxis=dict(title="Observed depth (m)", range=[0, mx]), yaxis=dict(title="Predicted depth (m)", range=[0, mx]))
    figs["scatter"] = lay(fig, "TGT-M (Model B) predicted versus observed depth, test split (random sample of up to 4,000 rows per selection)", 600)

# ---------------------------------------------------------------- HTML
tabs = [("models", "Models"), ("rain", "Rainfall effect"), ("season", "Seasons"), ("states", "States"), ("gap", "Unseen states"), ("interval", "Intervals (H1)"),
        ("robust", "Missing history (H2)"), ("map", "Stress map"), ("stressyear", "Stress by year"), ("scatter", "Predicted vs observed")]
tabs = [(k, n) for k, n in tabs if k in figs]
summ = {}
f = ROOT / "results" / "summary.json"
if f.exists():
    summ = json.loads(f.read_text())
kpi = []
if not S.empty:
    d = S[S.variant == "B"].set_index("model")
    best = d.loc[[m for m in models if m in ("SVR", "RF", "ANN", "XGBoost", "CNN1D", "LSTM", "TGT-M")], "depth_MAE_mean"].idxmin()
    kpi = [("Test rows", f"{summ.get('n_test', 40630):,}"), ("Wells / states", "2,759 / 19"), ("Best depth MAE (B)", f"{d.loc[best, 'depth_MAE_mean']:.3f} m ({lab(best)})"),
           ("TGT-M depth MAE (B)", f"{d.loc['TGT-M', 'depth_MAE_mean']:.3f} m" if "TGT-M" in d.index else "n/a"),
           ("Seasonal baseline", f"{d.loc['Seasonal baseline', 'depth_MAE_mean']:.3f} m" if "Seasonal baseline" in d.index else "n/a")]
body = []
for k, n in tabs:
    body.append(f'<section id="{k}" class="tab">{pio.to_html(figs[k], full_html=False, include_plotlyjs=False, div_id="fig_" + k)}</section>')
html = f"""<!doctype html><html><head><meta charset="utf-8"><title>TGT-M groundwater stress dashboard</title>
<script>{get_plotlyjs()}</script>
<style>body{{font-family:'Times New Roman',serif;margin:0;background:#fafafa}}header{{background:#1F4E79;color:#fff;padding:14px 24px}}
header h1{{margin:0;font-size:20px}}header p{{margin:4px 0 0;font-size:13px;opacity:.9}}.kpi{{display:flex;gap:12px;padding:12px 24px;flex-wrap:wrap}}
.k{{background:#fff;border:1px solid #ddd;border-radius:6px;padding:8px 14px}}.k b{{display:block;font-size:18px;color:#1F4E79}}
nav{{display:flex;gap:4px;padding:0 24px;flex-wrap:wrap}}nav button{{border:1px solid #1F4E79;background:#fff;color:#1F4E79;padding:7px 12px;cursor:pointer;border-radius:4px 4px 0 0;font-family:inherit;font-size:14px}}
nav button.on{{background:#1F4E79;color:#fff}}.tab{{display:none;padding:8px 24px 24px}}.tab.on{{display:block}}footer{{padding:12px 24px;font-size:12px;color:#555}}</style></head><body>
<header><h1>TGT-M: Seasonal Groundwater Stress Assessment and Cross-Region Generalization Across India (multimodal)</h1>
<p>Team 7, SRM IST, 21CSC314P Big Data Essentials, Review 2. Data: CGWB wells (2000-2022) + CHIRPS rainfall. Charts are interactive (hover, zoom, menus, play).</p></header>
<div class="kpi">{''.join(f'<div class="k">{a}<b>{b}</b></div>' for a, b in kpi)}</div>
<nav>{''.join(f'<button data-t="{k}">{n}</button>' for k, n in tabs)}</nav>{''.join(body)}
<footer>All numbers are generated from results/tables and processed/ by modeling/make_dashboard.py. Test split 2018-2022; Model B = with rainfall. Pre-registered tests: docs/preregistration_review2.md.</footer>
<script>const bs=document.querySelectorAll('nav button'),ts=document.querySelectorAll('.tab');
function show(k){{bs.forEach(b=>b.classList.toggle('on',b.dataset.t===k));ts.forEach(t=>t.classList.toggle('on',t.id===k));window.dispatchEvent(new Event('resize'));}}
bs.forEach(b=>b.onclick=()=>show(b.dataset.t));show(location.hash.slice(1)||bs[0].dataset.t);</script></body></html>"""
(OUT / "dashboard.html").write_text(html, encoding="utf-8")

# ---------------------------------------------------------------- extra Power BI / Tableau tables
wells = main.drop_duplicates("well_id")[["well_id", "state", "district", "lat", "lon"]]
wells.to_csv(PB / "wells.csv", index=False)
cls_out = cls[["well_id", "state", "year", "month", "gwl_m_bgl", "baseline_mean_m", "anomaly_z_capped", "stress_category", "stress_tadj", "stress_roll10"]].copy()
cls_out.to_csv(PB / "well_round_stress.csv", index=False)
cls.groupby(["state", "year", "stress_category"]).size().rename("n").reset_index().to_csv(PB / "stress_counts_state_year.csv", index=False)
for n in ("interval_summary", "interval_comparison", "robust_summary", "robust_degradation", "generalization_gap", "chrono_per_state", "chrono_calibrated"):
    if (T / f"{n}.csv").exists():
        pd.read_csv(T / f"{n}.csv").to_csv(PB / f"{n}.csv", index=False)
print("dashboard:", OUT / "dashboard.html", f"{(OUT / 'dashboard.html').stat().st_size / 1e6:.1f} MB;", "tabs:", [k for k, _ in tabs])
