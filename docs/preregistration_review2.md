# Pre-registration of two design-claim tests for TGT-M

Written 2026-10-12 18:35 IST, **before** any of the experiments below were run. Repository HEAD `c675583` (main);
fingerprint of `results/tables/*.csv` at that time: `fb7985cef8afa571`. The main comparison (chronological and
unseen-state tables) was already complete and is not changed by these tests.

Why these tests: in the main comparison TGT-M did not have the lowest point-prediction error (rank 5 of 7 on test depth MAE).
Its design makes two other claims that the point-error tables do not test: (H1) it gives usable uncertainty intervals, and
(H2) it is robust when recent history is missing. Both are tested below with all rules fixed in advance. A result against
TGT-M will be reported with the same prominence as one in its favour. These are the only two additional tests; no
other metric will be added after seeing results.

## Common protocol
- Model B (with rainfall) of every model, same eval-eligible test rows (2018-22, 40,630 rows), same target (capped anomaly z).
- Same best configuration per model as chosen on validation in the main comparison; models are re-trained with seeds 0, 1, 2
  (the retrained p = 0 numbers are reported again and replace the earlier ones only in this section).
- Uncertainty is quantified with paired bootstrap over wells (2,000 resamples, 95% intervals), as in the main comparison.
- Nothing is tuned on the test split. Calibration (conformal) uses the validation split only.

## H1: interval quality
Target nominal coverage 80% (10%-90%). Interval score (Gneiting and Raftery, alpha = 0.2):
IS = width + (2/alpha) * (lower - y)+ + (2/alpha) * (y - upper)+. Lower is better.

Competitors (all produce an 80% interval for the anomaly):
1. TGT-M native 10%/90% quantile heads, and TGT-M after conformalised quantile regression (CQR) on validation.
2. XGBoost quantile regression (q = 0.1, 0.9) native, and after the same CQR step. (Heteroscedastic baseline, primary comparator.)
3. Split-conformal constant-width intervals around the point prediction of every other model (SVR, RF, ANN, XGBoost, 1D-CNN, LSTM):
   half-width = 80% quantile of absolute validation residuals.

**H1 is supported** if TGT-M (native or CQR-calibrated, whichever is stated in advance as: the same variant is compared on both sides,
i.e. native vs native and CQR vs CQR) has a significantly lower mean interval score than XGBoost quantile regression, and its
empirical coverage lies within 80% +/- 3 points. If TGT-M is not significantly better, H1 is reported as not supported.

## H2: robustness to missing recent history
At test time only, a random share p of the observed past readings in the 12-round window is removed (p = 0.2, 0.4, 0.6; masks are
identical for every model; mask seed fixed per p). The features derived from removed rounds are set to missing: the sequence entries,
previous-round reading, same-season readings one and two years earlier, previous change and the last observation (recomputed from the
remaining window). Aggregate history features (baseline, running max/min, trend, counts) are left unchanged. Training is unchanged
(models are trained on the natural data).

Metric: anomaly MAE at each p, and the degradation D = MAE(p = 0.6) - MAE(p = 0).

**H2a (primary)**: D of TGT-M is significantly smaller than D of LSTM and of 1D-CNN (the other sequence models with identical inputs).
**H2b (secondary)**: TGT-M has the lowest anomaly MAE at p = 0.6 among all models.
Each is reported as supported or not supported on the paired bootstrap; tabular models are compared descriptively.

## Not allowed after seeing results
Changing architecture, hyperparameters, masks, p values, metrics, the primary comparators or the support criteria.

## Interpretation note (added before any result of these tests was available)
The wording of H1 allows two readings. The strict reading is the one that will be applied: H1 is **supported only if** TGT-M has a
significantly lower mean interval score than XGBoost quantile regression in **both** comparisons (native versus native, and
CQR-calibrated versus CQR-calibrated) **and** the native TGT-M coverage lies within 80% +/- 3 points. If it holds in only one of the two,
H1 is reported as partially supported / not supported, and both results are shown. For H2, "significantly" means the 95% paired-bootstrap
interval of the difference of degradations excludes zero.
