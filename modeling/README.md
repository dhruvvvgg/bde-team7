# Modeling and evaluation (Review 2)

Existing models (RF, SVR, ANN, XGBoost, 1D-CNN, LSTM) versus the proposed **TGT-M** (GRU-D + Temporal Fusion
Transformer), each with Model A (no rainfall) and Model B (with rainfall). Inputs are class-A columns of
`processed/gw_features_by_state/` only (see `docs/handoff_for_modeling.md`). Target: capped anomaly z of the hidden reading.

| File | Purpose |
|---|---|
| `data.py` | Builds model arrays: 42 (A) / 66 (B) tabular inputs, a (rows x 13 x 10) gap-aware sequence tensor, train-only scalers |
| `nets.py` | ANN, LSTM, 1D-CNN and TGT-M (GRU-D, VSN, GRN, gated attention, quantile heads) |
| `lib.py` | Training, prediction, metrics, hyperparameter grids (3-4 configs per model) |
| `run_chrono.py <model>` | Train 2000-15, tune on 2016-17 (seed 0), refit 3 seeds, test 2018-22 |
| `tgtm_job.py`, `tgtm_merge.py` | TGT-M as parallel 1-thread jobs (same protocol; pick best config on validation) |
| `run_geo.py <model> <loso\|g5>` | Time-respecting hold-out of states (19-fold LOSO or grouped 5-fold) |
| `aggregate.py` | Tables, paired bootstrap over wells, validation-calibrated class thresholds -> `results/tables/` |
| `export_powerbi.py` | Flat CSVs for Power BI / Tableau -> `results/powerbi/` |

Run order: `run_chrono.py` for each model, `run_geo.py`, `aggregate.py`, `export_powerbi.py`.
On Windows import `torch` before pandas/pyarrow (the DLL load order matters).

Leakage rules: no lat/lon/state/district or calendar-year inputs; depth lags are expressed relative to the past-only baseline;
scalers are fitted on training rows; the sequence window holds only earlier rounds plus the current round's season and
rainfall (rainfall ends at month M-1); the test split is never used to choose anything.

## Pre-registered tests (docs/preregistration_review2.md)
`robust_job.py <model> <seed>` retrains Model B and predicts under history masking (p = 0, 0.2, 0.4, 0.6) -> `results/robust/`;
`robust_eval.py` (H2, degradation) and `interval_eval.py` (H1, 80% interval score vs XGBoost quantile regression and conformal intervals);
`gap_test.py` is the post-hoc seen-to-unseen robustness comparison (reported as post hoc).
