# Results (Review 2)

- `chrono/`: per-seed predictions and metrics, chronological split (train 2000-15, val 2016-17, test 2018-22). `jobs/` holds the parallel TGT-M runs.
- `geo/`: time-respecting hold-out of states (`g5` grouped 5-fold for all models, `loso` 19 folds for XGBoost and SVR).
- `tables/`: tidy CSVs from `modeling/aggregate.py` (summary, calibrated classes, bootstrap, per season/state/completeness).
- `tables/model_metrics_*.csv`: full metric set per model (regression on depth and anomaly, classification overall and per class; mean of 3 seeds plus `_sd`), from `modeling/metrics_full.py`. Log loss is not computed because the models output numbers, not class probabilities.
- `powerbi/`: flat CSVs for Power BI / Tableau (includes `test_predictions.csv`).
- `summary.json`: tuning logs, selected configs, TGT-M diagnostics (80% interval coverage, VSN weights, attention).
- `logs/`: training logs. Regenerate tables with `python modeling/aggregate.py` and `python modeling/export_powerbi.py`.
- `robust/`: retrained predictions for the two pre-registered tests (H1 intervals, H2 missing history); tables `interval_*.csv`, `robust_*.csv`, `generalization_gap.csv` (post hoc).
