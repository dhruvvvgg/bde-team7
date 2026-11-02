# Retrain on Spark: old (pandas table) vs new (Spark table)

Old: `results/tables`; new: `results_spark/tables`. Same protocol, seeds, grids, splits and metrics; only the two columns `gw_max_to_date_m`, `gw_min_to_date_m` (and their normalised copies `*_rz`) differ in the model inputs (see `INPUT_DIFF.md`).

Flags: **NOISE** = |diff| <= 2 x pooled per-seed sd of the two runs (3 seeds; for rules with no sd: 0.01 on MAE-type, 0.005 on the others); **MOVED** = larger than that. Conclusion changes are collected in the first section.

## 0. Stated conclusions: old vs new

| item | old | new | changed? |
|---|---|---|---|
| TGT-M rank on depth MAE, Model B | 5 of 7 | 3 of 7 | **CHANGES CONCLUSION** |
| Model ranking by depth MAE, Model B | LSTM < CNN1D < XGBoost < ANN < TGT-M < RF < SVR | XGBoost < CNN1D < TGT-M < LSTM < ANN < RF < SVR | **CHANGES CONCLUSION** |
| TGT-M rank on depth MAE, Model A | 6 of 7 | 3 of 7 | **CHANGES CONCLUSION** |
| Model ranking by depth MAE, Model A | RF < XGBoost < LSTM < ANN < CNN1D < TGT-M < SVR | RF < CNN1D < TGT-M < LSTM < XGBoost < ANN < SVR | **CHANGES CONCLUSION** |
| Significance: TGT-M(B) - ANN(B) [depth_MAE] | CI [-0.0004, 0.0076] | CI [-0.0093, -0.0012] | **CHANGES CONCLUSION** |
| Significance: TGT-M(B) - ANN(B) [anom_MAE] | CI [0.0011, 0.0048] | CI [-0.0011, 0.0025] | **CHANGES CONCLUSION** |
| Significance: TGT-M(B) - CNN1D(B) [depth_MAE] | CI [0.0119, 0.0191] | CI [-0.0007, 0.0072] | **CHANGES CONCLUSION** |
| Significance: TGT-M(B) - LSTM(B) [depth_MAE] | CI [0.0213, 0.0288] | CI [-0.0065, 0.0012] | **CHANGES CONCLUSION** |
| Seen-to-unseen gap significance: TGT-M minus SVR / gap diff | CI [-0.0021, 0.0027] | CI [0.0072, 0.0125] | **CHANGES CONCLUSION** |
| Seen-to-unseen gap significance: TGT-M minus RF / gap diff | CI [-0.0118, -0.0078] | CI [-0.0042, 0.0003] | **CHANGES CONCLUSION** |
| Seen-to-unseen gap significance: TGT-M minus XGBoost / gap diff | CI [-0.0097, -0.0059] | CI [-0.0034, 0.0008] | **CHANGES CONCLUSION** |
| H1 (strict pre-registered rule) | SUPPORTED (native vs native -0.0496 [-0.0647, -0.0342], CQR vs CQR -0.0526 [-0.0681, -0.0368], native coverage 79.2%) | NOT supported (native vs native +0.0605 [0.0469, 0.0747], CQR vs CQR +0.0659 [0.0518, 0.0807], native coverage 79.5%) | **CHANGES CONCLUSION** |
| H2 CI significance: TGT-M minus ANN / difference of MAE at p=0.6 | [-0.0053, 0.0010] | [-0.0109, -0.0052] | **CHANGES CONCLUSION** |
| H2 CI significance: TGT-M minus XGBoost / difference of MAE at p=0.6 | [-0.0065, 0.0007] | [-0.0078, -0.0012] | **CHANGES CONCLUSION** |
| H2 CI significance: TGT-M minus CNN1D / difference of MAE at p=0.6 | [-0.0029, 0.0039] | [0.0002, 0.0062] | **CHANGES CONCLUSION** |
| H2a (D of TGT-M significantly smaller than LSTM and CNN1D) | SUPPORTED (vs LSTM: -0.0088 [-0.0118, -0.0058]; vs CNN1D: -0.0066 [-0.0096, -0.0036]) | SUPPORTED (vs LSTM: -0.0121 [-0.0149, -0.0093]; vs CNN1D: -0.0056 [-0.0082, -0.0030]) | no |
| H2b (TGT-M lowest anomaly MAE at p=0.6) | NOT supported (lowest at p=0.6: RF (0.8362); TGT-M 0.8450) | NOT supported (lowest at p=0.6: CNN1D (0.8259); TGT-M 0.8292) | no |
| Label sensitivity counts (threshold sets) | see G2 | see G2 | no |
| Stress label counts | {'Normal': 135523, 'Insufficient history': 52952, 'No reading': 34570, 'Watch': 12661, 'High Stress': 10609, 'Moderate Stress': 7513} | {'Normal': 135523, 'Insufficient history': 52952, 'No reading': 34570, 'Watch': 12661, 'High Stress': 10609, 'Moderate Stress': 7513} | no |

**15 of 19 checked conclusions changed.**

Table III: 140 numbers compared; 33 moved by more than seed noise:

| model   | variant   | metric          |    old |    new |    diff | seed sd (old/new)   |
|:--------|:----------|:----------------|-------:|-------:|--------:|:--------------------|
| RF      | A         | depth_MAE_mean  | 1.2505 | 1.2566 |  0.0061 | 0.0007 / 0.0007     |
| RF      | A         | depth_RMSE_mean | 1.9750 | 1.9851 |  0.0101 | 0.0007 / 0.0012     |
| RF      | A         | depth_R2_mean   | 0.8315 | 0.8298 | -0.0017 | 0.0001 / 0.0002     |
| RF      | A         | anom_MAE_mean   | 0.8334 | 0.8373 |  0.0039 | 0.0005 / 0.0010     |
| RF      | A         | anom_R2_mean    | 0.1947 | 0.1881 | -0.0066 | 0.0007 / 0.0017     |
| RF      | A         | macro_F1_mean   | 0.3128 | 0.3187 |  0.0059 | 0.0011 / 0.0013     |
| RF      | B         | depth_MAE_mean  | 1.1501 | 1.1481 | -0.0020 | 0.0003 / 0.0005     |
| RF      | B         | anom_MAE_mean   | 0.7767 | 0.7758 | -0.0009 | 0.0003 / 0.0001     |
| RF      | B         | anom_R2_mean    | 0.2634 | 0.2646 |  0.0011 | 0.0003 / 0.0005     |
| ANN     | A         | anom_R2_mean    | 0.1849 | 0.1372 | -0.0478 | 0.0268 / 0.0203     |
| XGBoost | A         | depth_MAE_mean  | 1.2636 | 1.2857 |  0.0221 | 0.0037 / 0.0027     |
| XGBoost | A         | depth_RMSE_mean | 2.0038 | 2.0433 |  0.0395 | 0.0061 / 0.0056     |
| XGBoost | A         | depth_R2_mean   | 0.8266 | 0.8197 | -0.0069 | 0.0011 / 0.0010     |
| XGBoost | A         | anom_MAE_mean   | 0.8353 | 0.8493 |  0.0140 | 0.0018 / 0.0008     |
| XGBoost | A         | anom_R2_mean    | 0.1872 | 0.1628 | -0.0244 | 0.0023 / 0.0028     |
| XGBoost | A         | acc_mean        | 0.8265 | 0.8153 | -0.0112 | 0.0001 / 0.0025     |
| XGBoost | A         | macro_F1_mean   | 0.3344 | 0.3413 |  0.0069 | 0.0028 / 0.0021     |
| XGBoost | B         | depth_MAE_mean  | 1.1211 | 1.1122 | -0.0089 | 0.0009 / 0.0014     |
| XGBoost | B         | depth_RMSE_mean | 1.8274 | 1.8204 | -0.0070 | 0.0004 / 0.0029     |
| XGBoost | B         | depth_R2_mean   | 0.8558 | 0.8569 |  0.0011 | 0.0001 / 0.0005     |
| XGBoost | B         | anom_MAE_mean   | 0.7585 | 0.7545 | -0.0040 | 0.0008 / 0.0008     |
| XGBoost | B         | anom_R2_mean    | 0.2774 | 0.2799 |  0.0024 | 0.0013 / 0.0007     |
| LSTM    | B         | depth_MAE_mean  | 1.1038 | 1.1259 |  0.0221 | 0.0027 / 0.0084     |
| LSTM    | B         | depth_RMSE_mean | 1.7843 | 1.8096 |  0.0252 | 0.0061 / 0.0087     |
| LSTM    | B         | depth_R2_mean   | 0.8625 | 0.8586 | -0.0039 | 0.0009 / 0.0014     |
| LSTM    | B         | anom_MAE_mean   | 0.7480 | 0.7579 |  0.0099 | 0.0017 / 0.0039     |
| LSTM    | B         | anom_R2_mean    | 0.2966 | 0.2893 | -0.0074 | 0.0035 / 0.0028     |
| TGT-M   | A         | depth_MAE_mean  | 1.3378 | 1.2741 | -0.0637 | 0.0079 / 0.0225     |
| TGT-M   | A         | depth_RMSE_mean | 2.1166 | 2.0301 | -0.0865 | 0.0040 / 0.0508     |
| TGT-M   | A         | depth_R2_mean   | 0.8065 | 0.8219 |  0.0154 | 0.0007 / 0.0089     |
| TGT-M   | A         | anom_MAE_mean   | 0.8796 | 0.8383 | -0.0413 | 0.0101 / 0.0056     |
| TGT-M   | A         | anom_R2_mean    | 0.1110 | 0.1761 |  0.0652 | 0.0229 / 0.0059     |
| TGT-M   | A         | acc_mean        | 0.7995 | 0.8157 |  0.0162 | 0.0090 / 0.0059     |

## A. Main results (Table III): chronological split, test 2018-2022, mean over 3 seeds

### A1. Ranking on test depth MAE (7 learners, Model B and Model A)

| variant   | TGT-M rank old   | TGT-M rank new   | order old (best to worst)                       | order new                                       | same order   |
|:----------|:-----------------|:-----------------|:------------------------------------------------|:------------------------------------------------|:-------------|
| B         | 5 of 7           | 3 of 7           | LSTM < CNN1D < XGBoost < ANN < TGT-M < RF < SVR | XGBoost < CNN1D < TGT-M < LSTM < ANN < RF < SVR | False        |
| A         | 6 of 7           | 3 of 7           | RF < XGBoost < LSTM < ANN < CNN1D < TGT-M < SVR | RF < CNN1D < TGT-M < LSTM < XGBoost < ANN < SVR | False        |

### A2. Every metric

| model             | variant   | metric          |     old |     new |    diff | seed sd (old/new)   | flag   |
|:------------------|:----------|:----------------|--------:|--------:|--------:|:--------------------|:-------|
| Prev. round       | A         | depth_MAE_mean  |  2.0502 |  2.0502 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | depth_RMSE_mean |  3.1140 |  3.1140 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | depth_R2_mean   |  0.5812 |  0.5812 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | anom_MAE_mean   |  1.4969 |  1.4969 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | anom_R2_mean    | -1.7310 | -1.7310 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | acc_mean        |  0.6983 |  0.6983 |  0.0000 |                     | NOISE  |
| Prev. round       | A         | macro_F1_mean   |  0.3222 |  0.3222 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | depth_MAE_mean  |  2.0502 |  2.0502 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | depth_RMSE_mean |  3.1140 |  3.1140 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | depth_R2_mean   |  0.5812 |  0.5812 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | anom_MAE_mean   |  1.4969 |  1.4969 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | anom_R2_mean    | -1.7310 | -1.7310 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | acc_mean        |  0.6983 |  0.6983 |  0.0000 |                     | NOISE  |
| Prev. round       | B         | macro_F1_mean   |  0.3222 |  0.3222 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | depth_MAE_mean  |  1.5070 |  1.5070 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | depth_RMSE_mean |  2.5557 |  2.5557 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | depth_R2_mean   |  0.7179 |  0.7179 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | anom_MAE_mean   |  0.9809 |  0.9809 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | anom_R2_mean    | -0.1745 | -0.1745 | -0.0000 |                     | NOISE  |
| Same season LY    | A         | acc_mean        |  0.7719 |  0.7719 |  0.0000 |                     | NOISE  |
| Same season LY    | A         | macro_F1_mean   |  0.3454 |  0.3454 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | depth_MAE_mean  |  1.5070 |  1.5070 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | depth_RMSE_mean |  2.5557 |  2.5557 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | depth_R2_mean   |  0.7179 |  0.7179 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | anom_MAE_mean   |  0.9809 |  0.9809 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | anom_R2_mean    | -0.1745 | -0.1745 | -0.0000 |                     | NOISE  |
| Same season LY    | B         | acc_mean        |  0.7719 |  0.7719 |  0.0000 |                     | NOISE  |
| Same season LY    | B         | macro_F1_mean   |  0.3454 |  0.3454 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | depth_MAE_mean  |  1.5484 |  1.5484 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | depth_RMSE_mean |  2.4453 |  2.4453 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | depth_R2_mean   |  0.7418 |  0.7418 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | anom_MAE_mean   |  0.9674 |  0.9674 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | anom_R2_mean    | -0.0173 | -0.0173 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | acc_mean        |  0.8412 |  0.8412 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A         | macro_F1_mean   |  0.2284 |  0.2284 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | depth_MAE_mean  |  1.5484 |  1.5484 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | depth_RMSE_mean |  2.4453 |  2.4453 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | depth_R2_mean   |  0.7418 |  0.7418 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | anom_MAE_mean   |  0.9674 |  0.9674 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | anom_R2_mean    | -0.0173 | -0.0173 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | acc_mean        |  0.8412 |  0.8412 |  0.0000 |                     | NOISE  |
| Seasonal baseline | B         | macro_F1_mean   |  0.2284 |  0.2284 |  0.0000 |                     | NOISE  |
| SVR               | A         | depth_MAE_mean  |  1.4589 |  1.4589 | -0.0000 | 0.0040 / 0.0022     | NOISE  |
| SVR               | A         | depth_RMSE_mean |  2.2480 |  2.2511 |  0.0031 | 0.0080 / 0.0016     | NOISE  |
| SVR               | A         | depth_R2_mean   |  0.7817 |  0.7812 | -0.0006 | 0.0015 / 0.0003     | NOISE  |
| SVR               | A         | anom_MAE_mean   |  0.9453 |  0.9451 | -0.0001 | 0.0007 / 0.0019     | NOISE  |
| SVR               | A         | anom_R2_mean    |  0.0351 |  0.0350 | -0.0001 | 0.0010 / 0.0028     | NOISE  |
| SVR               | A         | acc_mean        |  0.7873 |  0.7880 |  0.0007 | 0.0007 / 0.0013     | NOISE  |
| SVR               | A         | macro_F1_mean   |  0.3368 |  0.3391 |  0.0023 | 0.0014 / 0.0011     | NOISE  |
| SVR               | B         | depth_MAE_mean  |  1.1799 |  1.1789 | -0.0010 | 0.0067 / 0.0074     | NOISE  |
| SVR               | B         | depth_RMSE_mean |  1.9013 |  1.9018 |  0.0005 | 0.0040 / 0.0055     | NOISE  |
| SVR               | B         | depth_R2_mean   |  0.8439 |  0.8438 | -0.0001 | 0.0007 / 0.0009     | NOISE  |
| SVR               | B         | anom_MAE_mean   |  0.7904 |  0.7898 | -0.0006 | 0.0037 / 0.0045     | NOISE  |
| SVR               | B         | anom_R2_mean    |  0.2498 |  0.2504 |  0.0005 | 0.0037 / 0.0049     | NOISE  |
| SVR               | B         | acc_mean        |  0.8321 |  0.8321 |  0.0000 | 0.0002 / 0.0005     | NOISE  |
| SVR               | B         | macro_F1_mean   |  0.3125 |  0.3128 |  0.0003 | 0.0024 / 0.0023     | NOISE  |
| RF                | A         | depth_MAE_mean  |  1.2505 |  1.2566 |  0.0061 | 0.0007 / 0.0007     | MOVED  |
| RF                | A         | depth_RMSE_mean |  1.9750 |  1.9851 |  0.0101 | 0.0007 / 0.0012     | MOVED  |
| RF                | A         | depth_R2_mean   |  0.8315 |  0.8298 | -0.0017 | 0.0001 / 0.0002     | MOVED  |
| RF                | A         | anom_MAE_mean   |  0.8334 |  0.8373 |  0.0039 | 0.0005 / 0.0010     | MOVED  |
| RF                | A         | anom_R2_mean    |  0.1947 |  0.1881 | -0.0066 | 0.0007 / 0.0017     | MOVED  |
| RF                | A         | acc_mean        |  0.8252 |  0.8249 | -0.0003 | 0.0004 / 0.0009     | NOISE  |
| RF                | A         | macro_F1_mean   |  0.3128 |  0.3187 |  0.0059 | 0.0011 / 0.0013     | MOVED  |
| RF                | B         | depth_MAE_mean  |  1.1501 |  1.1481 | -0.0020 | 0.0003 / 0.0005     | MOVED  |
| RF                | B         | depth_RMSE_mean |  1.8488 |  1.8513 |  0.0025 | 0.0008 / 0.0020     | NOISE  |
| RF                | B         | depth_R2_mean   |  0.8524 |  0.8520 | -0.0004 | 0.0001 / 0.0003     | NOISE  |
| RF                | B         | anom_MAE_mean   |  0.7767 |  0.7758 | -0.0009 | 0.0003 / 0.0001     | MOVED  |
| RF                | B         | anom_R2_mean    |  0.2634 |  0.2646 |  0.0011 | 0.0003 / 0.0005     | MOVED  |
| RF                | B         | acc_mean        |  0.8363 |  0.8358 | -0.0005 | 0.0004 / 0.0004     | NOISE  |
| RF                | B         | macro_F1_mean   |  0.3132 |  0.3117 | -0.0016 | 0.0016 / 0.0017     | NOISE  |
| ANN               | A         | depth_MAE_mean  |  1.2704 |  1.3146 |  0.0442 | 0.0254 / 0.0295     | NOISE  |
| ANN               | A         | depth_RMSE_mean |  2.0212 |  2.0695 |  0.0483 | 0.0399 / 0.0270     | NOISE  |
| ANN               | A         | depth_R2_mean   |  0.8235 |  0.8150 | -0.0085 | 0.0070 / 0.0048     | NOISE  |
| ANN               | A         | anom_MAE_mean   |  0.8376 |  0.8682 |  0.0306 | 0.0159 / 0.0156     | NOISE  |
| ANN               | A         | anom_R2_mean    |  0.1849 |  0.1372 | -0.0478 | 0.0268 / 0.0203     | MOVED  |
| ANN               | A         | acc_mean        |  0.8235 |  0.8059 | -0.0175 | 0.0124 / 0.0041     | NOISE  |
| ANN               | A         | macro_F1_mean   |  0.3244 |  0.3454 |  0.0210 | 0.0252 / 0.0056     | NOISE  |
| ANN               | B         | depth_MAE_mean  |  1.1249 |  1.1286 |  0.0037 | 0.0175 / 0.0013     | NOISE  |
| ANN               | B         | depth_RMSE_mean |  1.8177 |  1.8252 |  0.0075 | 0.0222 / 0.0074     | NOISE  |
| ANN               | B         | depth_R2_mean   |  0.8573 |  0.8561 | -0.0012 | 0.0035 / 0.0012     | NOISE  |
| ANN               | B         | anom_MAE_mean   |  0.7597 |  0.7611 |  0.0014 | 0.0069 / 0.0009     | NOISE  |
| ANN               | B         | anom_R2_mean    |  0.2832 |  0.2824 | -0.0009 | 0.0057 / 0.0024     | NOISE  |
| ANN               | B         | acc_mean        |  0.8366 |  0.8348 | -0.0018 | 0.0025 / 0.0014     | NOISE  |
| ANN               | B         | macro_F1_mean   |  0.3273 |  0.3312 |  0.0039 | 0.0078 / 0.0074     | NOISE  |
| XGBoost           | A         | depth_MAE_mean  |  1.2636 |  1.2857 |  0.0221 | 0.0037 / 0.0027     | MOVED  |
| XGBoost           | A         | depth_RMSE_mean |  2.0038 |  2.0433 |  0.0395 | 0.0061 / 0.0056     | MOVED  |
| XGBoost           | A         | depth_R2_mean   |  0.8266 |  0.8197 | -0.0069 | 0.0011 / 0.0010     | MOVED  |
| XGBoost           | A         | anom_MAE_mean   |  0.8353 |  0.8493 |  0.0140 | 0.0018 / 0.0008     | MOVED  |
| XGBoost           | A         | anom_R2_mean    |  0.1872 |  0.1628 | -0.0244 | 0.0023 / 0.0028     | MOVED  |
| XGBoost           | A         | acc_mean        |  0.8265 |  0.8153 | -0.0112 | 0.0001 / 0.0025     | MOVED  |
| XGBoost           | A         | macro_F1_mean   |  0.3344 |  0.3413 |  0.0069 | 0.0028 / 0.0021     | MOVED  |
| XGBoost           | B         | depth_MAE_mean  |  1.1211 |  1.1122 | -0.0089 | 0.0009 / 0.0014     | MOVED  |
| XGBoost           | B         | depth_RMSE_mean |  1.8274 |  1.8204 | -0.0070 | 0.0004 / 0.0029     | MOVED  |
| XGBoost           | B         | depth_R2_mean   |  0.8558 |  0.8569 |  0.0011 | 0.0001 / 0.0005     | MOVED  |
| XGBoost           | B         | anom_MAE_mean   |  0.7585 |  0.7545 | -0.0040 | 0.0008 / 0.0008     | MOVED  |
| XGBoost           | B         | anom_R2_mean    |  0.2774 |  0.2799 |  0.0024 | 0.0013 / 0.0007     | MOVED  |
| XGBoost           | B         | acc_mean        |  0.8375 |  0.8378 |  0.0003 | 0.0005 / 0.0003     | NOISE  |
| XGBoost           | B         | macro_F1_mean   |  0.3293 |  0.3325 |  0.0032 | 0.0014 / 0.0034     | NOISE  |
| CNN1D             | A         | depth_MAE_mean  |  1.2871 |  1.2732 | -0.0139 | 0.0109 / 0.0136     | NOISE  |
| CNN1D             | A         | depth_RMSE_mean |  2.0450 |  2.0339 | -0.0111 | 0.0207 / 0.0068     | NOISE  |
| CNN1D             | A         | depth_R2_mean   |  0.8194 |  0.8213 |  0.0020 | 0.0037 / 0.0012     | NOISE  |
| CNN1D             | A         | anom_MAE_mean   |  0.8439 |  0.8356 | -0.0083 | 0.0084 / 0.0108     | NOISE  |
| CNN1D             | A         | anom_R2_mean    |  0.1724 |  0.1833 |  0.0108 | 0.0152 / 0.0145     | NOISE  |
| CNN1D             | A         | acc_mean        |  0.8193 |  0.8207 |  0.0014 | 0.0088 / 0.0061     | NOISE  |
| CNN1D             | A         | macro_F1_mean   |  0.3527 |  0.3551 |  0.0023 | 0.0056 / 0.0028     | NOISE  |
| CNN1D             | B         | depth_MAE_mean  |  1.1132 |  1.1200 |  0.0068 | 0.0189 / 0.0093     | NOISE  |
| CNN1D             | B         | depth_RMSE_mean |  1.8074 |  1.8092 |  0.0018 | 0.0290 / 0.0096     | NOISE  |
| CNN1D             | B         | depth_R2_mean   |  0.8589 |  0.8586 | -0.0003 | 0.0045 / 0.0015     | NOISE  |
| CNN1D             | B         | anom_MAE_mean   |  0.7507 |  0.7529 |  0.0022 | 0.0065 / 0.0043     | NOISE  |
| CNN1D             | B         | anom_R2_mean    |  0.2934 |  0.2921 | -0.0013 | 0.0062 / 0.0061     | NOISE  |
| CNN1D             | B         | acc_mean        |  0.8369 |  0.8373 |  0.0004 | 0.0012 / 0.0035     | NOISE  |
| CNN1D             | B         | macro_F1_mean   |  0.3423 |  0.3409 | -0.0014 | 0.0044 / 0.0086     | NOISE  |
| LSTM              | A         | depth_MAE_mean  |  1.2686 |  1.2852 |  0.0166 | 0.0322 / 0.0435     | NOISE  |
| LSTM              | A         | depth_RMSE_mean |  2.0012 |  2.0380 |  0.0368 | 0.0374 / 0.0667     | NOISE  |
| LSTM              | A         | depth_R2_mean   |  0.8270 |  0.8205 | -0.0065 | 0.0065 / 0.0118     | NOISE  |
| LSTM              | A         | anom_MAE_mean   |  0.8405 |  0.8467 |  0.0062 | 0.0206 / 0.0253     | NOISE  |
| LSTM              | A         | anom_R2_mean    |  0.1778 |  0.1648 | -0.0130 | 0.0281 / 0.0437     | NOISE  |
| LSTM              | A         | acc_mean        |  0.8151 |  0.8097 | -0.0055 | 0.0111 / 0.0190     | NOISE  |
| LSTM              | A         | macro_F1_mean   |  0.3485 |  0.3419 | -0.0066 | 0.0045 / 0.0099     | NOISE  |
| LSTM              | B         | depth_MAE_mean  |  1.1038 |  1.1259 |  0.0221 | 0.0027 / 0.0084     | MOVED  |
| LSTM              | B         | depth_RMSE_mean |  1.7843 |  1.8096 |  0.0252 | 0.0061 / 0.0087     | MOVED  |
| LSTM              | B         | depth_R2_mean   |  0.8625 |  0.8586 | -0.0039 | 0.0009 / 0.0014     | MOVED  |
| LSTM              | B         | anom_MAE_mean   |  0.7480 |  0.7579 |  0.0099 | 0.0017 / 0.0039     | MOVED  |
| LSTM              | B         | anom_R2_mean    |  0.2966 |  0.2893 | -0.0074 | 0.0035 / 0.0028     | MOVED  |
| LSTM              | B         | acc_mean        |  0.8366 |  0.8370 |  0.0004 | 0.0007 / 0.0010     | NOISE  |
| LSTM              | B         | macro_F1_mean   |  0.3400 |  0.3339 | -0.0061 | 0.0029 / 0.0064     | NOISE  |
| TGT-M             | A         | depth_MAE_mean  |  1.3378 |  1.2741 | -0.0637 | 0.0079 / 0.0225     | MOVED  |
| TGT-M             | A         | depth_RMSE_mean |  2.1166 |  2.0301 | -0.0865 | 0.0040 / 0.0508     | MOVED  |
| TGT-M             | A         | depth_R2_mean   |  0.8065 |  0.8219 |  0.0154 | 0.0007 / 0.0089     | MOVED  |
| TGT-M             | A         | anom_MAE_mean   |  0.8796 |  0.8383 | -0.0413 | 0.0101 / 0.0056     | MOVED  |
| TGT-M             | A         | anom_R2_mean    |  0.1110 |  0.1761 |  0.0652 | 0.0229 / 0.0059     | MOVED  |
| TGT-M             | A         | acc_mean        |  0.7995 |  0.8157 |  0.0162 | 0.0090 / 0.0059     | MOVED  |
| TGT-M             | A         | macro_F1_mean   |  0.3446 |  0.3375 | -0.0071 | 0.0159 / 0.0163     | NOISE  |
| TGT-M             | B         | depth_MAE_mean  |  1.1287 |  1.1233 | -0.0053 | 0.0083 / 0.0061     | NOISE  |
| TGT-M             | B         | depth_RMSE_mean |  1.8111 |  1.8166 |  0.0055 | 0.0099 / 0.0089     | NOISE  |
| TGT-M             | B         | depth_R2_mean   |  0.8583 |  0.8575 | -0.0009 | 0.0015 / 0.0014     | NOISE  |
| TGT-M             | B         | anom_MAE_mean   |  0.7627 |  0.7618 | -0.0009 | 0.0032 / 0.0025     | NOISE  |
| TGT-M             | B         | anom_R2_mean    |  0.2791 |  0.2775 | -0.0016 | 0.0035 / 0.0038     | NOISE  |
| TGT-M             | B         | acc_mean        |  0.8308 |  0.8292 | -0.0016 | 0.0021 / 0.0018     | NOISE  |
| TGT-M             | B         | macro_F1_mean   |  0.3388 |  0.3482 |  0.0094 | 0.0059 / 0.0034     | NOISE  |

## B. Classification (default thresholds 1/1.5/2 and validation-calibrated)

| model             | variant   | thresholds            | metric            |    old |    new |    diff | seed sd (old/new)   | flag   |
|:------------------|:----------|:----------------------|:------------------|-------:|-------:|--------:|:--------------------|:-------|
| Prev. round       | A/B       | default 1/1.5/2       | accuracy          | 0.6983 | 0.6983 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | default 1/1.5/2       | F1_macro          | 0.3222 | 0.3222 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | default 1/1.5/2       | balanced_accuracy | 0.3533 | 0.3533 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | default 1/1.5/2       | MCC               | 0.1486 | 0.1486 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | default 1/1.5/2       | AUC_ROC_stressed  | 0.6801 | 0.6801 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | default 1/1.5/2       | AUC_ROC_High      | 0.6814 | 0.6814 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | accuracy          | 0.6815 | 0.6815 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | F1_macro          | 0.3252 | 0.3252 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | balanced_accuracy | 0.3618 | 0.3618 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | MCC               | 0.1548 | 0.1548 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | AUC_ROC_stressed  | 0.6801 | 0.6801 |  0.0000 |                     | NOISE  |
| Prev. round       | A/B       | validation-calibrated | AUC_ROC_High      | 0.6814 | 0.6814 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | accuracy          | 0.7719 | 0.7719 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | F1_macro          | 0.3454 | 0.3454 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | balanced_accuracy | 0.3423 | 0.3423 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | MCC               | 0.1758 | 0.1758 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | AUC_ROC_stressed  | 0.6783 | 0.6783 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | default 1/1.5/2       | AUC_ROC_High      | 0.6614 | 0.6614 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | accuracy          | 0.7585 | 0.7585 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | F1_macro          | 0.3427 | 0.3427 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | balanced_accuracy | 0.3445 | 0.3445 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | MCC               | 0.1741 | 0.1741 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | AUC_ROC_stressed  | 0.6783 | 0.6783 |  0.0000 |                     | NOISE  |
| Same season LY    | A/B       | validation-calibrated | AUC_ROC_High      | 0.6614 | 0.6614 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | accuracy          | 0.8412 | 0.8412 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | F1_macro          | 0.2284 | 0.2284 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | balanced_accuracy | 0.2500 | 0.2500 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | MCC               | 0.0000 | 0.0000 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | AUC_ROC_stressed  | 0.5000 | 0.5000 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | default 1/1.5/2       | AUC_ROC_High      | 0.5000 | 0.5000 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | accuracy          | 0.8412 | 0.8412 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | F1_macro          | 0.2284 | 0.2284 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | balanced_accuracy | 0.2500 | 0.2500 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | MCC               | 0.0000 | 0.0000 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | AUC_ROC_stressed  | 0.5000 | 0.5000 |  0.0000 |                     | NOISE  |
| Seasonal baseline | A/B       | validation-calibrated | AUC_ROC_High      | 0.5000 | 0.5000 |  0.0000 |                     | NOISE  |
| SVR               | A         | default 1/1.5/2       | accuracy          | 0.7873 | 0.7880 |  0.0007 | 0.0007 / 0.0013     | NOISE  |
| SVR               | A         | default 1/1.5/2       | F1_macro          | 0.3368 | 0.3391 |  0.0023 | 0.0014 / 0.0011     | NOISE  |
| SVR               | A         | default 1/1.5/2       | balanced_accuracy | 0.3282 | 0.3300 |  0.0019 | 0.0010 / 0.0009     | NOISE  |
| SVR               | A         | default 1/1.5/2       | MCC               | 0.1896 | 0.1914 |  0.0018 | 0.0011 / 0.0008     | NOISE  |
| SVR               | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7360 | 0.7371 |  0.0011 | 0.0003 / 0.0003     | MOVED  |
| SVR               | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7320 | 0.7331 |  0.0011 | 0.0005 / 0.0004     | MOVED  |
| SVR               | A         | validation-calibrated | accuracy          | 0.6711 | 0.6986 |  0.0275 | 0.0262 / 0.0401     | NOISE  |
| SVR               | A         | validation-calibrated | F1_macro          | 0.3327 | 0.3362 |  0.0035 | 0.0080 / 0.0083     | NOISE  |
| SVR               | A         | validation-calibrated | balanced_accuracy | 0.3639 | 0.3571 | -0.0068 | 0.0030 / 0.0077     | NOISE  |
| SVR               | A         | validation-calibrated | MCC               | 0.1791 | 0.1866 |  0.0075 | 0.0066 / 0.0096     | NOISE  |
| SVR               | A         | validation-calibrated | AUC_ROC_stressed  | 0.7360 | 0.7371 |  0.0011 | 0.0003 / 0.0003     | MOVED  |
| SVR               | A         | validation-calibrated | AUC_ROC_High      | 0.7320 | 0.7331 |  0.0011 | 0.0005 / 0.0004     | MOVED  |
| SVR               | B         | default 1/1.5/2       | accuracy          | 0.8321 | 0.8321 |  0.0000 | 0.0002 / 0.0005     | NOISE  |
| SVR               | B         | default 1/1.5/2       | F1_macro          | 0.3125 | 0.3128 |  0.0003 | 0.0024 / 0.0023     | NOISE  |
| SVR               | B         | default 1/1.5/2       | balanced_accuracy | 0.3004 | 0.3006 |  0.0002 | 0.0017 / 0.0019     | NOISE  |
| SVR               | B         | default 1/1.5/2       | MCC               | 0.2020 | 0.2027 |  0.0007 | 0.0027 / 0.0040     | NOISE  |
| SVR               | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7809 | 0.7819 |  0.0010 | 0.0006 / 0.0007     | NOISE  |
| SVR               | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7731 | 0.7742 |  0.0011 | 0.0002 / 0.0004     | MOVED  |
| SVR               | B         | validation-calibrated | accuracy          | 0.7794 | 0.7709 | -0.0085 | 0.0127 / 0.0120     | NOISE  |
| SVR               | B         | validation-calibrated | F1_macro          | 0.3738 | 0.3741 |  0.0003 | 0.0028 / 0.0019     | NOISE  |
| SVR               | B         | validation-calibrated | balanced_accuracy | 0.3688 | 0.3729 |  0.0041 | 0.0019 / 0.0034     | NOISE  |
| SVR               | B         | validation-calibrated | MCC               | 0.2433 | 0.2426 | -0.0007 | 0.0051 / 0.0068     | NOISE  |
| SVR               | B         | validation-calibrated | AUC_ROC_stressed  | 0.7809 | 0.7819 |  0.0010 | 0.0006 / 0.0007     | NOISE  |
| SVR               | B         | validation-calibrated | AUC_ROC_High      | 0.7731 | 0.7742 |  0.0011 | 0.0002 / 0.0004     | MOVED  |
| RF                | A         | default 1/1.5/2       | accuracy          | 0.8252 | 0.8249 | -0.0003 | 0.0004 / 0.0009     | NOISE  |
| RF                | A         | default 1/1.5/2       | F1_macro          | 0.3128 | 0.3187 |  0.0059 | 0.0011 / 0.0013     | MOVED  |
| RF                | A         | default 1/1.5/2       | balanced_accuracy | 0.3004 | 0.3047 |  0.0043 | 0.0009 / 0.0007     | MOVED  |
| RF                | A         | default 1/1.5/2       | MCC               | 0.1832 | 0.1895 |  0.0063 | 0.0029 / 0.0025     | MOVED  |
| RF                | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7603 | 0.7578 | -0.0024 | 0.0003 / 0.0003     | MOVED  |
| RF                | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7513 | 0.7502 | -0.0010 | 0.0004 / 0.0003     | MOVED  |
| RF                | A         | validation-calibrated | accuracy          | 0.7671 | 0.7611 | -0.0060 | 0.0022 / 0.0013     | MOVED  |
| RF                | A         | validation-calibrated | F1_macro          | 0.3601 | 0.3595 | -0.0007 | 0.0028 / 0.0026     | NOISE  |
| RF                | A         | validation-calibrated | balanced_accuracy | 0.3617 | 0.3616 | -0.0002 | 0.0009 / 0.0034     | NOISE  |
| RF                | A         | validation-calibrated | MCC               | 0.2165 | 0.2121 | -0.0043 | 0.0004 / 0.0014     | MOVED  |
| RF                | A         | validation-calibrated | AUC_ROC_stressed  | 0.7603 | 0.7578 | -0.0024 | 0.0003 / 0.0003     | MOVED  |
| RF                | A         | validation-calibrated | AUC_ROC_High      | 0.7513 | 0.7502 | -0.0010 | 0.0004 / 0.0003     | MOVED  |
| RF                | B         | default 1/1.5/2       | accuracy          | 0.8363 | 0.8358 | -0.0005 | 0.0004 / 0.0004     | NOISE  |
| RF                | B         | default 1/1.5/2       | F1_macro          | 0.3132 | 0.3117 | -0.0016 | 0.0016 / 0.0017     | NOISE  |
| RF                | B         | default 1/1.5/2       | balanced_accuracy | 0.2994 | 0.2982 | -0.0012 | 0.0011 / 0.0011     | NOISE  |
| RF                | B         | default 1/1.5/2       | MCC               | 0.2030 | 0.2000 | -0.0030 | 0.0023 / 0.0026     | NOISE  |
| RF                | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7809 | 0.7796 | -0.0013 | 0.0008 / 0.0002     | MOVED  |
| RF                | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7675 | 0.7671 | -0.0004 | 0.0007 / 0.0001     | NOISE  |
| RF                | B         | validation-calibrated | accuracy          | 0.7757 | 0.7678 | -0.0079 | 0.0137 / 0.0018     | NOISE  |
| RF                | B         | validation-calibrated | F1_macro          | 0.3704 | 0.3708 |  0.0004 | 0.0021 / 0.0010     | NOISE  |
| RF                | B         | validation-calibrated | balanced_accuracy | 0.3682 | 0.3707 |  0.0025 | 0.0073 / 0.0014     | NOISE  |
| RF                | B         | validation-calibrated | MCC               | 0.2425 | 0.2387 | -0.0038 | 0.0012 / 0.0010     | MOVED  |
| RF                | B         | validation-calibrated | AUC_ROC_stressed  | 0.7809 | 0.7796 | -0.0013 | 0.0008 / 0.0002     | MOVED  |
| RF                | B         | validation-calibrated | AUC_ROC_High      | 0.7675 | 0.7671 | -0.0004 | 0.0007 / 0.0001     | NOISE  |
| ANN               | A         | default 1/1.5/2       | accuracy          | 0.8235 | 0.8059 | -0.0175 | 0.0124 / 0.0041     | NOISE  |
| ANN               | A         | default 1/1.5/2       | F1_macro          | 0.3244 | 0.3454 |  0.0210 | 0.0252 / 0.0056     | NOISE  |
| ANN               | A         | default 1/1.5/2       | balanced_accuracy | 0.3114 | 0.3297 |  0.0182 | 0.0225 / 0.0051     | NOISE  |
| ANN               | A         | default 1/1.5/2       | MCC               | 0.1999 | 0.2026 |  0.0027 | 0.0208 / 0.0109     | NOISE  |
| ANN               | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7556 | 0.7527 | -0.0029 | 0.0077 / 0.0012     | NOISE  |
| ANN               | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7453 | 0.7442 | -0.0011 | 0.0077 / 0.0026     | NOISE  |
| ANN               | A         | validation-calibrated | accuracy          | 0.7509 | 0.7390 | -0.0119 | 0.0273 / 0.0073     | NOISE  |
| ANN               | A         | validation-calibrated | F1_macro          | 0.3540 | 0.3526 | -0.0013 | 0.0050 / 0.0059     | NOISE  |
| ANN               | A         | validation-calibrated | balanced_accuracy | 0.3606 | 0.3643 |  0.0037 | 0.0043 / 0.0056     | NOISE  |
| ANN               | A         | validation-calibrated | MCC               | 0.2140 | 0.2062 | -0.0078 | 0.0106 / 0.0036     | NOISE  |
| ANN               | A         | validation-calibrated | AUC_ROC_stressed  | 0.7556 | 0.7527 | -0.0029 | 0.0077 / 0.0012     | NOISE  |
| ANN               | A         | validation-calibrated | AUC_ROC_High      | 0.7453 | 0.7442 | -0.0011 | 0.0077 / 0.0026     | NOISE  |
| ANN               | B         | default 1/1.5/2       | accuracy          | 0.8366 | 0.8348 | -0.0018 | 0.0025 / 0.0014     | NOISE  |
| ANN               | B         | default 1/1.5/2       | F1_macro          | 0.3273 | 0.3312 |  0.0039 | 0.0078 / 0.0074     | NOISE  |
| ANN               | B         | default 1/1.5/2       | balanced_accuracy | 0.3088 | 0.3120 |  0.0032 | 0.0064 / 0.0059     | NOISE  |
| ANN               | B         | default 1/1.5/2       | MCC               | 0.2198 | 0.2216 |  0.0018 | 0.0061 / 0.0079     | NOISE  |
| ANN               | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7933 | 0.7932 | -0.0001 | 0.0011 / 0.0009     | NOISE  |
| ANN               | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7828 | 0.7811 | -0.0017 | 0.0029 / 0.0009     | NOISE  |
| ANN               | B         | validation-calibrated | accuracy          | 0.7774 | 0.7780 |  0.0006 | 0.0068 / 0.0116     | NOISE  |
| ANN               | B         | validation-calibrated | F1_macro          | 0.3807 | 0.3803 | -0.0003 | 0.0017 / 0.0009     | NOISE  |
| ANN               | B         | validation-calibrated | balanced_accuracy | 0.3792 | 0.3787 | -0.0005 | 0.0026 / 0.0038     | NOISE  |
| ANN               | B         | validation-calibrated | MCC               | 0.2596 | 0.2570 | -0.0026 | 0.0046 / 0.0035     | NOISE  |
| ANN               | B         | validation-calibrated | AUC_ROC_stressed  | 0.7933 | 0.7932 | -0.0001 | 0.0011 / 0.0009     | NOISE  |
| ANN               | B         | validation-calibrated | AUC_ROC_High      | 0.7828 | 0.7811 | -0.0017 | 0.0029 / 0.0009     | NOISE  |
| XGBoost           | A         | default 1/1.5/2       | accuracy          | 0.8265 | 0.8153 | -0.0112 | 0.0001 / 0.0025     | MOVED  |
| XGBoost           | A         | default 1/1.5/2       | F1_macro          | 0.3344 | 0.3413 |  0.0069 | 0.0028 / 0.0021     | MOVED  |
| XGBoost           | A         | default 1/1.5/2       | balanced_accuracy | 0.3179 | 0.3253 |  0.0075 | 0.0016 / 0.0018     | MOVED  |
| XGBoost           | A         | default 1/1.5/2       | MCC               | 0.2163 | 0.2148 | -0.0015 | 0.0020 / 0.0026     | NOISE  |
| XGBoost           | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7635 | 0.7613 | -0.0022 | 0.0007 / 0.0016     | NOISE  |
| XGBoost           | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7508 | 0.7494 | -0.0015 | 0.0009 / 0.0017     | NOISE  |
| XGBoost           | A         | validation-calibrated | accuracy          | 0.7441 | 0.7364 | -0.0076 | 0.0154 / 0.0056     | NOISE  |
| XGBoost           | A         | validation-calibrated | F1_macro          | 0.3514 | 0.3470 | -0.0045 | 0.0053 / 0.0011     | NOISE  |
| XGBoost           | A         | validation-calibrated | balanced_accuracy | 0.3630 | 0.3597 | -0.0033 | 0.0029 / 0.0009     | NOISE  |
| XGBoost           | A         | validation-calibrated | MCC               | 0.2149 | 0.2067 | -0.0082 | 0.0042 / 0.0013     | MOVED  |
| XGBoost           | A         | validation-calibrated | AUC_ROC_stressed  | 0.7635 | 0.7613 | -0.0022 | 0.0007 / 0.0016     | NOISE  |
| XGBoost           | A         | validation-calibrated | AUC_ROC_High      | 0.7508 | 0.7494 | -0.0015 | 0.0009 / 0.0017     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | accuracy          | 0.8375 | 0.8378 |  0.0003 | 0.0005 / 0.0003     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | F1_macro          | 0.3293 | 0.3325 |  0.0032 | 0.0014 / 0.0034     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | balanced_accuracy | 0.3103 | 0.3124 |  0.0021 | 0.0010 / 0.0023     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | MCC               | 0.2230 | 0.2256 |  0.0026 | 0.0018 / 0.0027     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7884 | 0.7884 | -0.0001 | 0.0006 / 0.0001     | NOISE  |
| XGBoost           | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7736 | 0.7740 |  0.0004 | 0.0007 / 0.0003     | NOISE  |
| XGBoost           | B         | validation-calibrated | accuracy          | 0.7745 | 0.7671 | -0.0073 | 0.0113 / 0.0008     | NOISE  |
| XGBoost           | B         | validation-calibrated | F1_macro          | 0.3682 | 0.3650 | -0.0032 | 0.0007 / 0.0005     | MOVED  |
| XGBoost           | B         | validation-calibrated | balanced_accuracy | 0.3657 | 0.3655 | -0.0002 | 0.0051 / 0.0008     | NOISE  |
| XGBoost           | B         | validation-calibrated | MCC               | 0.2409 | 0.2381 | -0.0028 | 0.0024 / 0.0011     | NOISE  |
| XGBoost           | B         | validation-calibrated | AUC_ROC_stressed  | 0.7884 | 0.7884 | -0.0001 | 0.0006 / 0.0001     | NOISE  |
| XGBoost           | B         | validation-calibrated | AUC_ROC_High      | 0.7736 | 0.7740 |  0.0004 | 0.0007 / 0.0003     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | accuracy          | 0.8193 | 0.8207 |  0.0014 | 0.0088 / 0.0061     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | F1_macro          | 0.3527 | 0.3551 |  0.0023 | 0.0056 / 0.0028     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | balanced_accuracy | 0.3319 | 0.3333 |  0.0014 | 0.0071 / 0.0021     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | MCC               | 0.2182 | 0.2223 |  0.0041 | 0.0081 / 0.0087     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7682 | 0.7688 |  0.0006 | 0.0012 / 0.0042     | NOISE  |
| CNN1D             | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7555 | 0.7572 |  0.0017 | 0.0019 / 0.0039     | NOISE  |
| CNN1D             | A         | validation-calibrated | accuracy          | 0.7360 | 0.7042 | -0.0318 | 0.0099 / 0.0292     | NOISE  |
| CNN1D             | A         | validation-calibrated | F1_macro          | 0.3606 | 0.3495 | -0.0112 | 0.0014 / 0.0097     | NOISE  |
| CNN1D             | A         | validation-calibrated | balanced_accuracy | 0.3718 | 0.3740 |  0.0022 | 0.0031 / 0.0032     | NOISE  |
| CNN1D             | A         | validation-calibrated | MCC               | 0.2187 | 0.2108 | -0.0079 | 0.0047 / 0.0114     | NOISE  |
| CNN1D             | A         | validation-calibrated | AUC_ROC_stressed  | 0.7682 | 0.7688 |  0.0006 | 0.0012 / 0.0042     | NOISE  |
| CNN1D             | A         | validation-calibrated | AUC_ROC_High      | 0.7555 | 0.7572 |  0.0017 | 0.0019 / 0.0039     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | accuracy          | 0.8369 | 0.8373 |  0.0004 | 0.0012 / 0.0035     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | F1_macro          | 0.3423 | 0.3409 | -0.0014 | 0.0044 / 0.0086     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | balanced_accuracy | 0.3198 | 0.3183 | -0.0015 | 0.0020 / 0.0071     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | MCC               | 0.2369 | 0.2366 | -0.0003 | 0.0032 / 0.0071     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7970 | 0.7991 |  0.0021 | 0.0045 / 0.0011     | NOISE  |
| CNN1D             | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7831 | 0.7855 |  0.0023 | 0.0053 / 0.0014     | NOISE  |
| CNN1D             | B         | validation-calibrated | accuracy          | 0.7823 | 0.7843 |  0.0020 | 0.0077 / 0.0091     | NOISE  |
| CNN1D             | B         | validation-calibrated | F1_macro          | 0.3787 | 0.3827 |  0.0040 | 0.0031 / 0.0039     | NOISE  |
| CNN1D             | B         | validation-calibrated | balanced_accuracy | 0.3747 | 0.3779 |  0.0032 | 0.0058 / 0.0001     | NOISE  |
| CNN1D             | B         | validation-calibrated | MCC               | 0.2608 | 0.2653 |  0.0046 | 0.0008 / 0.0056     | NOISE  |
| CNN1D             | B         | validation-calibrated | AUC_ROC_stressed  | 0.7970 | 0.7991 |  0.0021 | 0.0045 / 0.0011     | NOISE  |
| CNN1D             | B         | validation-calibrated | AUC_ROC_High      | 0.7831 | 0.7855 |  0.0023 | 0.0053 / 0.0014     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | accuracy          | 0.8151 | 0.8097 | -0.0055 | 0.0111 / 0.0190     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | F1_macro          | 0.3485 | 0.3419 | -0.0066 | 0.0045 / 0.0099     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | balanced_accuracy | 0.3293 | 0.3258 | -0.0035 | 0.0068 / 0.0125     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | MCC               | 0.2110 | 0.2003 | -0.0106 | 0.0025 / 0.0114     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7676 | 0.7613 | -0.0062 | 0.0036 / 0.0054     | NOISE  |
| LSTM              | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7573 | 0.7492 | -0.0081 | 0.0033 / 0.0040     | MOVED  |
| LSTM              | A         | validation-calibrated | accuracy          | 0.7143 | 0.7208 |  0.0065 | 0.0450 / 0.0166     | NOISE  |
| LSTM              | A         | validation-calibrated | F1_macro          | 0.3536 | 0.3524 | -0.0012 | 0.0150 / 0.0047     | NOISE  |
| LSTM              | A         | validation-calibrated | balanced_accuracy | 0.3741 | 0.3736 | -0.0005 | 0.0029 / 0.0077     | NOISE  |
| LSTM              | A         | validation-calibrated | MCC               | 0.2134 | 0.2096 | -0.0038 | 0.0177 / 0.0046     | NOISE  |
| LSTM              | A         | validation-calibrated | AUC_ROC_stressed  | 0.7676 | 0.7613 | -0.0062 | 0.0036 / 0.0054     | NOISE  |
| LSTM              | A         | validation-calibrated | AUC_ROC_High      | 0.7573 | 0.7492 | -0.0081 | 0.0033 / 0.0040     | MOVED  |
| LSTM              | B         | default 1/1.5/2       | accuracy          | 0.8366 | 0.8370 |  0.0004 | 0.0007 / 0.0010     | NOISE  |
| LSTM              | B         | default 1/1.5/2       | F1_macro          | 0.3400 | 0.3339 | -0.0061 | 0.0029 / 0.0064     | NOISE  |
| LSTM              | B         | default 1/1.5/2       | balanced_accuracy | 0.3179 | 0.3134 | -0.0045 | 0.0020 / 0.0051     | NOISE  |
| LSTM              | B         | default 1/1.5/2       | MCC               | 0.2361 | 0.2298 | -0.0063 | 0.0029 / 0.0085     | NOISE  |
| LSTM              | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.8006 | 0.7975 | -0.0031 | 0.0022 / 0.0021     | NOISE  |
| LSTM              | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7881 | 0.7841 | -0.0039 | 0.0032 / 0.0034     | NOISE  |
| LSTM              | B         | validation-calibrated | accuracy          | 0.7872 | 0.7871 | -0.0001 | 0.0007 / 0.0083     | NOISE  |
| LSTM              | B         | validation-calibrated | F1_macro          | 0.3879 | 0.3853 | -0.0026 | 0.0044 / 0.0038     | NOISE  |
| LSTM              | B         | validation-calibrated | balanced_accuracy | 0.3834 | 0.3801 | -0.0032 | 0.0050 / 0.0065     | NOISE  |
| LSTM              | B         | validation-calibrated | MCC               | 0.2670 | 0.2652 | -0.0018 | 0.0056 / 0.0026     | NOISE  |
| LSTM              | B         | validation-calibrated | AUC_ROC_stressed  | 0.8006 | 0.7975 | -0.0031 | 0.0022 / 0.0021     | NOISE  |
| LSTM              | B         | validation-calibrated | AUC_ROC_High      | 0.7881 | 0.7841 | -0.0039 | 0.0032 / 0.0034     | NOISE  |
| TGT-M             | A         | default 1/1.5/2       | accuracy          | 0.7995 | 0.8157 |  0.0162 | 0.0090 / 0.0059     | MOVED  |
| TGT-M             | A         | default 1/1.5/2       | F1_macro          | 0.3446 | 0.3375 | -0.0071 | 0.0159 / 0.0163     | NOISE  |
| TGT-M             | A         | default 1/1.5/2       | balanced_accuracy | 0.3318 | 0.3251 | -0.0067 | 0.0163 / 0.0101     | NOISE  |
| TGT-M             | A         | default 1/1.5/2       | MCC               | 0.1987 | 0.2123 |  0.0136 | 0.0168 / 0.0117     | NOISE  |
| TGT-M             | A         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7595 | 0.7618 |  0.0023 | 0.0033 / 0.0066     | NOISE  |
| TGT-M             | A         | default 1/1.5/2       | AUC_ROC_High      | 0.7516 | 0.7495 | -0.0021 | 0.0048 / 0.0084     | NOISE  |
| TGT-M             | A         | validation-calibrated | accuracy          | 0.7058 | 0.7538 |  0.0480 | 0.0233 / 0.0160     | MOVED  |
| TGT-M             | A         | validation-calibrated | F1_macro          | 0.3491 | 0.3612 |  0.0120 | 0.0078 / 0.0062     | NOISE  |
| TGT-M             | A         | validation-calibrated | balanced_accuracy | 0.3728 | 0.3662 | -0.0066 | 0.0056 / 0.0079     | NOISE  |
| TGT-M             | A         | validation-calibrated | MCC               | 0.2031 | 0.2186 |  0.0155 | 0.0061 / 0.0069     | MOVED  |
| TGT-M             | A         | validation-calibrated | AUC_ROC_stressed  | 0.7595 | 0.7618 |  0.0023 | 0.0033 / 0.0066     | NOISE  |
| TGT-M             | A         | validation-calibrated | AUC_ROC_High      | 0.7516 | 0.7495 | -0.0021 | 0.0048 / 0.0084     | NOISE  |
| TGT-M             | B         | default 1/1.5/2       | accuracy          | 0.8308 | 0.8292 | -0.0016 | 0.0021 / 0.0018     | NOISE  |
| TGT-M             | B         | default 1/1.5/2       | F1_macro          | 0.3388 | 0.3482 |  0.0094 | 0.0059 / 0.0034     | NOISE  |
| TGT-M             | B         | default 1/1.5/2       | balanced_accuracy | 0.3183 | 0.3260 |  0.0077 | 0.0045 / 0.0028     | MOVED  |
| TGT-M             | B         | default 1/1.5/2       | MCC               | 0.2255 | 0.2332 |  0.0077 | 0.0035 / 0.0041     | MOVED  |
| TGT-M             | B         | default 1/1.5/2       | AUC_ROC_stressed  | 0.7947 | 0.7901 | -0.0045 | 0.0006 / 0.0031     | NOISE  |
| TGT-M             | B         | default 1/1.5/2       | AUC_ROC_High      | 0.7830 | 0.7758 | -0.0072 | 0.0015 / 0.0047     | MOVED  |
| TGT-M             | B         | validation-calibrated | accuracy          | 0.7703 | 0.7721 |  0.0018 | 0.0049 / 0.0086     | NOISE  |
| TGT-M             | B         | validation-calibrated | F1_macro          | 0.3816 | 0.3792 | -0.0024 | 0.0048 / 0.0020     | NOISE  |
| TGT-M             | B         | validation-calibrated | balanced_accuracy | 0.3844 | 0.3805 | -0.0039 | 0.0086 / 0.0063     | NOISE  |
| TGT-M             | B         | validation-calibrated | MCC               | 0.2556 | 0.2528 | -0.0028 | 0.0051 / 0.0036     | NOISE  |
| TGT-M             | B         | validation-calibrated | AUC_ROC_stressed  | 0.7947 | 0.7901 | -0.0045 | 0.0006 / 0.0031     | NOISE  |
| TGT-M             | B         | validation-calibrated | AUC_ROC_High      | 0.7830 | 0.7758 | -0.0072 | 0.0015 / 0.0047     | MOVED  |

## C. Rainfall effect (Model B minus Model A) with 95% paired-bootstrap CI over wells

| comparison                       | metric    |   diff_old |   diff_new |   diff of estimate | old CI             | new CI             | sig old   | sig new   | flag                              |
|:---------------------------------|:----------|-----------:|-----------:|-------------------:|:-------------------|:-------------------|:----------|:----------|:----------------------------------|
| SVR: B - A (rainfall effect)     | depth_MAE |    -0.2790 |    -0.2800 |            -0.0009 | [-0.2923, -0.2661] | [-0.2936, -0.2670] | True      | True      | same                              |
| SVR: B - A (rainfall effect)     | anom_MAE  |    -0.1549 |    -0.1553 |            -0.0004 | [-0.1594, -0.1501] | [-0.1598, -0.1506] | True      | True      | same                              |
| RF: B - A (rainfall effect)      | depth_MAE |    -0.1004 |    -0.1085 |            -0.0081 | [-0.1084, -0.0925] | [-0.1168, -0.0999] | True      | True      | same                              |
| RF: B - A (rainfall effect)      | anom_MAE  |    -0.0567 |    -0.0615 |            -0.0048 | [-0.0602, -0.0531] | [-0.0651, -0.0578] | True      | True      | same                              |
| ANN: B - A (rainfall effect)     | depth_MAE |    -0.1455 |    -0.1860 |            -0.0405 | [-0.1540, -0.1369] | [-0.1955, -0.1763] | True      | True      | same                              |
| ANN: B - A (rainfall effect)     | anom_MAE  |    -0.0779 |    -0.1071 |            -0.0292 | [-0.0817, -0.0739] | [-0.1113, -0.1027] | True      | True      | same                              |
| XGBoost: B - A (rainfall effect) | depth_MAE |    -0.1424 |    -0.1735 |            -0.0310 | [-0.1518, -0.1329] | [-0.1837, -0.1630] | True      | True      | same                              |
| XGBoost: B - A (rainfall effect) | anom_MAE  |    -0.0768 |    -0.0948 |            -0.0180 | [-0.0808, -0.0730] | [-0.0990, -0.0907] | True      | True      | same                              |
| CNN1D: B - A (rainfall effect)   | depth_MAE |    -0.1739 |    -0.1532 |             0.0207 | [-0.1836, -0.1642] | [-0.1622, -0.1442] | True      | True      | same                              |
| CNN1D: B - A (rainfall effect)   | anom_MAE  |    -0.0932 |    -0.0827 |             0.0105 | [-0.0975, -0.0891] | [-0.0866, -0.0792] | True      | True      | same                              |
| LSTM: B - A (rainfall effect)    | depth_MAE |    -0.1649 |    -0.1593 |             0.0056 | [-0.1738, -0.1561] | [-0.1676, -0.1510] | True      | True      | same                              |
| LSTM: B - A (rainfall effect)    | anom_MAE  |    -0.0925 |    -0.0889 |             0.0037 | [-0.0964, -0.0888] | [-0.0924, -0.0854] | True      | True      | same                              |
| TGT-M: B - A (rainfall effect)   | depth_MAE |    -0.2091 |    -0.1508 |             0.0583 | [-0.2201, -0.1982] | [-0.1600, -0.1418] | True      | True      | same                              |
| TGT-M: B - A (rainfall effect)   | anom_MAE  |    -0.1169 |    -0.0765 |             0.0403 | [-0.1209, -0.1127] | [-0.0801, -0.0729] | True      | True      | same                              |
| TGT-M(B) - Prev. round(B)        | depth_MAE |    -0.9215 |    -0.9269 |            -0.0053 | [-0.9544, -0.8896] | [-0.9593, -0.8955] | True      | True      | same                              |
| TGT-M(B) - Prev. round(B)        | anom_MAE  |    -0.7342 |    -0.7351 |            -0.0009 | [-0.7571, -0.7119] | [-0.7578, -0.7126] | True      | True      | same                              |
| TGT-M(B) - Same season LY(B)     | depth_MAE |    -0.3783 |    -0.3837 |            -0.0053 | [-0.3988, -0.3579] | [-0.4043, -0.3639] | True      | True      | same                              |
| TGT-M(B) - Same season LY(B)     | anom_MAE  |    -0.2182 |    -0.2191 |            -0.0009 | [-0.2259, -0.2104] | [-0.2269, -0.2115] | True      | True      | same                              |
| TGT-M(B) - Seasonal baseline(B)  | depth_MAE |    -0.4197 |    -0.4251 |            -0.0053 | [-0.4496, -0.3928] | [-0.4553, -0.3973] | True      | True      | same                              |
| TGT-M(B) - Seasonal baseline(B)  | anom_MAE  |    -0.2047 |    -0.2056 |            -0.0009 | [-0.2141, -0.1949] | [-0.2153, -0.1958] | True      | True      | same                              |
| TGT-M(B) - SVR(B)                | depth_MAE |    -0.0512 |    -0.0556 |            -0.0044 | [-0.0612, -0.0427] | [-0.0659, -0.0465] | True      | True      | same                              |
| TGT-M(B) - SVR(B)                | anom_MAE  |    -0.0277 |    -0.0280 |            -0.0003 | [-0.0303, -0.0252] | [-0.0307, -0.0253] | True      | True      | same                              |
| TGT-M(B) - RF(B)                 | depth_MAE |    -0.0214 |    -0.0248 |            -0.0034 | [-0.0278, -0.0152] | [-0.0321, -0.0181] | True      | True      | same                              |
| TGT-M(B) - RF(B)                 | anom_MAE  |    -0.0140 |    -0.0140 |             0.0000 | [-0.0169, -0.0111] | [-0.0170, -0.0108] | True      | True      | same                              |
| TGT-M(B) - ANN(B)                | depth_MAE |     0.0037 |    -0.0053 |            -0.0090 | [-0.0004, 0.0076]  | [-0.0093, -0.0012] | False     | True      | CHANGES CONCLUSION (significance) |
| TGT-M(B) - ANN(B)                | anom_MAE  |     0.0030 |     0.0007 |            -0.0023 | [0.0011, 0.0048]   | [-0.0011, 0.0025]  | True      | False     | CHANGES CONCLUSION (significance) |
| TGT-M(B) - XGBoost(B)            | depth_MAE |     0.0075 |     0.0111 |             0.0036 | [0.0014, 0.0137]   | [0.0049, 0.0170]   | True      | True      | same                              |
| TGT-M(B) - XGBoost(B)            | anom_MAE  |     0.0042 |     0.0072 |             0.0030 | [0.0018, 0.0068]   | [0.0047, 0.0098]   | True      | True      | same                              |
| TGT-M(B) - CNN1D(B)              | depth_MAE |     0.0155 |     0.0033 |            -0.0122 | [0.0119, 0.0191]   | [-0.0007, 0.0072]  | True      | False     | CHANGES CONCLUSION (significance) |
| TGT-M(B) - CNN1D(B)              | anom_MAE  |     0.0120 |     0.0089 |            -0.0031 | [0.0101, 0.0139]   | [0.0069, 0.0108]   | True      | True      | same                              |
| TGT-M(B) - LSTM(B)               | depth_MAE |     0.0249 |    -0.0026 |            -0.0275 | [0.0213, 0.0288]   | [-0.0065, 0.0012]  | True      | False     | CHANGES CONCLUSION (significance) |
| TGT-M(B) - LSTM(B)               | anom_MAE  |     0.0147 |     0.0039 |            -0.0108 | [0.0131, 0.0164]   | [0.0022, 0.0057]   | True      | True      | same                              |

## D. Unseen-state (geographic) generalisation (single seed per cell, so no sd: bands 0.02 on MAE/RMSE, 0.01 otherwise)

| model   | scheme   | variant   | metric       |    old |    new |    diff | seed sd (old/new)   | flag   |
|:--------|:---------|:----------|:-------------|-------:|-------:|--------:|:--------------------|:-------|
| ANN     | g5       | A         | depth_MAE    | 1.2592 | 1.2609 |  0.0017 |                     | NOISE  |
| ANN     | g5       | A         | depth_RMSE   | 2.0184 | 2.0438 |  0.0254 |                     | MOVED  |
| ANN     | g5       | A         | anom_MAE     | 0.8395 | 0.8338 | -0.0057 |                     | NOISE  |
| ANN     | g5       | A         | anom_R2      | 0.1763 | 0.1837 |  0.0073 |                     | NOISE  |
| ANN     | g5       | A         | macro_F1     | 0.3365 | 0.3192 | -0.0172 |                     | MOVED  |
| ANN     | g5       | A         | AUC_stressed | 0.7560 | 0.7528 | -0.0032 |                     | NOISE  |
| ANN     | g5       | B         | depth_MAE    | 1.1389 | 1.1367 | -0.0022 |                     | NOISE  |
| ANN     | g5       | B         | depth_RMSE   | 1.8417 | 1.8467 |  0.0050 |                     | NOISE  |
| ANN     | g5       | B         | anom_MAE     | 0.7709 | 0.7692 | -0.0017 |                     | NOISE  |
| ANN     | g5       | B         | anom_R2      | 0.2681 | 0.2695 |  0.0015 |                     | NOISE  |
| ANN     | g5       | B         | macro_F1     | 0.3286 | 0.3198 | -0.0088 |                     | NOISE  |
| ANN     | g5       | B         | AUC_stressed | 0.7850 | 0.7857 |  0.0007 |                     | NOISE  |
| CNN1D   | g5       | A         | depth_MAE    | 1.2792 | 1.2512 | -0.0280 |                     | MOVED  |
| CNN1D   | g5       | A         | depth_RMSE   | 2.0311 | 2.0065 | -0.0246 |                     | MOVED  |
| CNN1D   | g5       | A         | anom_MAE     | 0.8455 | 0.8242 | -0.0213 |                     | MOVED  |
| CNN1D   | g5       | A         | anom_R2      | 0.1695 | 0.1967 |  0.0271 |                     | MOVED  |
| CNN1D   | g5       | A         | macro_F1     | 0.3415 | 0.3465 |  0.0051 |                     | NOISE  |
| CNN1D   | g5       | A         | AUC_stressed | 0.7629 | 0.7666 |  0.0037 |                     | NOISE  |
| CNN1D   | g5       | B         | depth_MAE    | 1.1235 | 1.1356 |  0.0120 |                     | NOISE  |
| CNN1D   | g5       | B         | depth_RMSE   | 1.8298 | 1.8379 |  0.0081 |                     | NOISE  |
| CNN1D   | g5       | B         | anom_MAE     | 0.7601 | 0.7628 |  0.0026 |                     | NOISE  |
| CNN1D   | g5       | B         | anom_R2      | 0.2803 | 0.2793 | -0.0009 |                     | NOISE  |
| CNN1D   | g5       | B         | macro_F1     | 0.3369 | 0.3376 |  0.0007 |                     | NOISE  |
| CNN1D   | g5       | B         | AUC_stressed | 0.7946 | 0.7909 | -0.0037 |                     | NOISE  |
| LSTM    | g5       | A         | depth_MAE    | 1.2586 | 1.2764 |  0.0178 |                     | NOISE  |
| LSTM    | g5       | A         | depth_RMSE   | 2.0071 | 2.0220 |  0.0149 |                     | NOISE  |
| LSTM    | g5       | A         | anom_MAE     | 0.8341 | 0.8436 |  0.0095 |                     | NOISE  |
| LSTM    | g5       | A         | anom_R2      | 0.1841 | 0.1734 | -0.0107 |                     | MOVED  |
| LSTM    | g5       | A         | macro_F1     | 0.3369 | 0.3406 |  0.0037 |                     | NOISE  |
| LSTM    | g5       | A         | AUC_stressed | 0.7642 | 0.7627 | -0.0015 |                     | NOISE  |
| LSTM    | g5       | B         | depth_MAE    | 1.1333 | 1.1599 |  0.0265 |                     | MOVED  |
| LSTM    | g5       | B         | depth_RMSE   | 1.8326 | 1.8654 |  0.0328 |                     | MOVED  |
| LSTM    | g5       | B         | anom_MAE     | 0.7634 | 0.7726 |  0.0092 |                     | NOISE  |
| LSTM    | g5       | B         | anom_R2      | 0.2753 | 0.2703 | -0.0049 |                     | NOISE  |
| LSTM    | g5       | B         | macro_F1     | 0.3442 | 0.3401 | -0.0041 |                     | NOISE  |
| LSTM    | g5       | B         | AUC_stressed | 0.7906 | 0.7857 | -0.0049 |                     | NOISE  |
| RF      | g5       | A         | depth_MAE    | 1.2668 | 1.2726 |  0.0058 |                     | NOISE  |
| RF      | g5       | A         | depth_RMSE   | 2.0056 | 2.0144 |  0.0088 |                     | NOISE  |
| RF      | g5       | A         | anom_MAE     | 0.8402 | 0.8442 |  0.0039 |                     | NOISE  |
| RF      | g5       | A         | anom_R2      | 0.1849 | 0.1779 | -0.0071 |                     | NOISE  |
| RF      | g5       | A         | macro_F1     | 0.3054 | 0.3148 |  0.0094 |                     | NOISE  |
| RF      | g5       | A         | AUC_stressed | 0.7552 | 0.7519 | -0.0033 |                     | NOISE  |
| RF      | g5       | B         | depth_MAE    | 1.1618 | 1.1609 | -0.0008 |                     | NOISE  |
| RF      | g5       | B         | depth_RMSE   | 1.8695 | 1.8735 |  0.0040 |                     | NOISE  |
| RF      | g5       | B         | anom_MAE     | 0.7826 | 0.7821 | -0.0004 |                     | NOISE  |
| RF      | g5       | B         | anom_R2      | 0.2572 | 0.2570 | -0.0002 |                     | NOISE  |
| RF      | g5       | B         | macro_F1     | 0.3077 | 0.3047 | -0.0030 |                     | NOISE  |
| RF      | g5       | B         | AUC_stressed | 0.7774 | 0.7754 | -0.0020 |                     | NOISE  |
| SVR     | g5       | A         | depth_MAE    | 1.4188 | 1.4216 |  0.0028 |                     | NOISE  |
| SVR     | g5       | A         | depth_RMSE   | 2.2143 | 2.2205 |  0.0062 |                     | NOISE  |
| SVR     | g5       | A         | anom_MAE     | 0.9200 | 0.9214 |  0.0014 |                     | NOISE  |
| SVR     | g5       | A         | anom_R2      | 0.0687 | 0.0663 | -0.0024 |                     | NOISE  |
| SVR     | g5       | A         | macro_F1     | 0.3302 | 0.3286 | -0.0016 |                     | NOISE  |
| SVR     | g5       | A         | AUC_stressed | 0.7341 | 0.7345 |  0.0003 |                     | NOISE  |
| SVR     | g5       | B         | depth_MAE    | 1.1853 | 1.1861 |  0.0008 |                     | NOISE  |
| SVR     | g5       | B         | depth_RMSE   | 1.9336 | 1.9344 |  0.0008 |                     | NOISE  |
| SVR     | g5       | B         | anom_MAE     | 0.7894 | 0.7896 |  0.0002 |                     | NOISE  |
| SVR     | g5       | B         | anom_R2      | 0.2473 | 0.2468 | -0.0005 |                     | NOISE  |
| SVR     | g5       | B         | macro_F1     | 0.3115 | 0.3116 |  0.0001 |                     | NOISE  |
| SVR     | g5       | B         | AUC_stressed | 0.7784 | 0.7786 |  0.0002 |                     | NOISE  |
| SVR     | loso     | A         | depth_MAE    | 1.4393 | 1.4459 |  0.0066 |                     | NOISE  |
| SVR     | loso     | A         | depth_RMSE   | 2.2332 | 2.2463 |  0.0132 |                     | NOISE  |
| SVR     | loso     | A         | anom_MAE     | 0.9341 | 0.9365 |  0.0024 |                     | NOISE  |
| SVR     | loso     | A         | anom_R2      | 0.0479 | 0.0441 | -0.0038 |                     | NOISE  |
| SVR     | loso     | A         | macro_F1     | 0.3304 | 0.3315 |  0.0011 |                     | NOISE  |
| SVR     | loso     | A         | AUC_stressed | 0.7326 | 0.7330 |  0.0003 |                     | NOISE  |
| SVR     | loso     | B         | depth_MAE    | 1.1879 | 1.1847 | -0.0033 |                     | NOISE  |
| SVR     | loso     | B         | depth_RMSE   | 1.9290 | 1.9224 | -0.0066 |                     | NOISE  |
| SVR     | loso     | B         | anom_MAE     | 0.7932 | 0.7917 | -0.0015 |                     | NOISE  |
| SVR     | loso     | B         | anom_R2      | 0.2435 | 0.2451 |  0.0017 |                     | NOISE  |
| SVR     | loso     | B         | macro_F1     | 0.3038 | 0.3027 | -0.0010 |                     | NOISE  |
| SVR     | loso     | B         | AUC_stressed | 0.7769 | 0.7777 |  0.0007 |                     | NOISE  |
| TGT-M   | g5       | A         | depth_MAE    | 1.2853 | 1.2403 | -0.0451 |                     | MOVED  |
| TGT-M   | g5       | A         | depth_RMSE   | 2.0428 | 1.9899 | -0.0529 |                     | MOVED  |
| TGT-M   | g5       | A         | anom_MAE     | 0.8495 | 0.8231 | -0.0264 |                     | MOVED  |
| TGT-M   | g5       | A         | anom_R2      | 0.1641 | 0.1938 |  0.0297 |                     | MOVED  |
| TGT-M   | g5       | A         | macro_F1     | 0.3396 | 0.3298 | -0.0098 |                     | NOISE  |
| TGT-M   | g5       | A         | AUC_stressed | 0.7608 | 0.7615 |  0.0008 |                     | NOISE  |
| TGT-M   | g5       | B         | depth_MAE    | 1.1265 | 1.1434 |  0.0170 |                     | NOISE  |
| TGT-M   | g5       | B         | depth_RMSE   | 1.8174 | 1.8499 |  0.0325 |                     | MOVED  |
| TGT-M   | g5       | B         | anom_MAE     | 0.7624 | 0.7688 |  0.0064 |                     | NOISE  |
| TGT-M   | g5       | B         | anom_R2      | 0.2766 | 0.2676 | -0.0089 |                     | NOISE  |
| TGT-M   | g5       | B         | macro_F1     | 0.3433 | 0.3422 | -0.0012 |                     | NOISE  |
| TGT-M   | g5       | B         | AUC_stressed | 0.7932 | 0.7854 | -0.0077 |                     | NOISE  |
| XGBoost | g5       | A         | depth_MAE    | 1.2810 | 1.3057 |  0.0246 |                     | MOVED  |
| XGBoost | g5       | A         | depth_RMSE   | 2.0398 | 2.0838 |  0.0440 |                     | MOVED  |
| XGBoost | g5       | A         | anom_MAE     | 0.8409 | 0.8568 |  0.0159 |                     | NOISE  |
| XGBoost | g5       | A         | anom_R2      | 0.1783 | 0.1509 | -0.0274 |                     | MOVED  |
| XGBoost | g5       | A         | macro_F1     | 0.3271 | 0.3320 |  0.0049 |                     | NOISE  |
| XGBoost | g5       | A         | AUC_stressed | 0.7581 | 0.7537 | -0.0043 |                     | NOISE  |
| XGBoost | g5       | B         | depth_MAE    | 1.1284 | 1.1217 | -0.0066 |                     | NOISE  |
| XGBoost | g5       | B         | depth_RMSE   | 1.8415 | 1.8391 | -0.0025 |                     | NOISE  |
| XGBoost | g5       | B         | anom_MAE     | 0.7625 | 0.7595 | -0.0030 |                     | NOISE  |
| XGBoost | g5       | B         | anom_R2      | 0.2728 | 0.2733 |  0.0005 |                     | NOISE  |
| XGBoost | g5       | B         | macro_F1     | 0.3234 | 0.3222 | -0.0013 |                     | NOISE  |
| XGBoost | g5       | B         | AUC_stressed | 0.7854 | 0.7858 |  0.0004 |                     | NOISE  |
| XGBoost | loso     | A         | depth_MAE    | 1.2763 | 1.2991 |  0.0227 |                     | MOVED  |
| XGBoost | loso     | A         | depth_RMSE   | 2.0254 | 2.0696 |  0.0442 |                     | MOVED  |
| XGBoost | loso     | A         | anom_MAE     | 0.8402 | 0.8538 |  0.0135 |                     | NOISE  |
| XGBoost | loso     | A         | anom_R2      | 0.1808 | 0.1572 | -0.0236 |                     | MOVED  |
| XGBoost | loso     | A         | macro_F1     | 0.3272 | 0.3332 |  0.0060 |                     | NOISE  |
| XGBoost | loso     | A         | AUC_stressed | 0.7603 | 0.7569 | -0.0035 |                     | NOISE  |
| XGBoost | loso     | B         | depth_MAE    | 1.1302 | 1.1207 | -0.0095 |                     | NOISE  |
| XGBoost | loso     | B         | depth_RMSE   | 1.8414 | 1.8363 | -0.0050 |                     | NOISE  |
| XGBoost | loso     | B         | anom_MAE     | 0.7629 | 0.7584 | -0.0045 |                     | NOISE  |
| XGBoost | loso     | B         | anom_R2      | 0.2725 | 0.2760 |  0.0035 |                     | NOISE  |
| XGBoost | loso     | B         | macro_F1     | 0.3240 | 0.3253 |  0.0013 |                     | NOISE  |
| XGBoost | loso     | B         | AUC_stressed | 0.7844 | 0.7870 |  0.0026 |                     | NOISE  |

### D2. Post-hoc seen-to-unseen gap (95% CI)

| comparison          | what     |   estimate_old |   estimate_new |   ci_lo_old |   ci_hi_old |   ci_lo_new |   ci_hi_new | sig old   | sig new   |
|:--------------------|:---------|---------------:|---------------:|------------:|------------:|------------:|------------:|:----------|:----------|
| SVR                 | own gap  |        -0.0043 |        -0.0053 |     -0.0065 |     -0.0021 |     -0.0076 |     -0.0031 | True      | True      |
| RF                  | own gap  |         0.0058 |         0.0065 |      0.0045 |      0.0071 |      0.0053 |      0.0077 | True      | True      |
| ANN                 | own gap  |         0.0036 |         0.0090 |      0.0015 |      0.0057 |      0.0070 |      0.0110 | True      | True      |
| XGBoost             | own gap  |         0.0038 |         0.0059 |      0.0028 |      0.0049 |      0.0047 |      0.0070 | True      | True      |
| CNN1D               | own gap  |         0.0160 |         0.0135 |      0.0141 |      0.0181 |      0.0114 |      0.0154 | True      | True      |
| LSTM                | own gap  |         0.0170 |         0.0168 |      0.0147 |      0.0191 |      0.0148 |      0.0188 | True      | True      |
| TGT-M               | own gap  |        -0.0040 |         0.0046 |     -0.0059 |     -0.0020 |      0.0025 |      0.0067 | True      | True      |
| TGT-M minus SVR     | gap diff |         0.0003 |         0.0098 |     -0.0021 |      0.0027 |      0.0072 |      0.0125 | False     | True      |
| TGT-M minus RF      | gap diff |        -0.0098 |        -0.0019 |     -0.0118 |     -0.0078 |     -0.0042 |      0.0003 | True      | False     |
| TGT-M minus ANN     | gap diff |        -0.0076 |        -0.0044 |     -0.0095 |     -0.0055 |     -0.0068 |     -0.0020 | True      | True      |
| TGT-M minus XGBoost | gap diff |        -0.0078 |        -0.0013 |     -0.0097 |     -0.0059 |     -0.0034 |      0.0008 | True      | False     |
| TGT-M minus CNN1D   | gap diff |        -0.0200 |        -0.0089 |     -0.0226 |     -0.0176 |     -0.0110 |     -0.0069 | True      | True      |
| TGT-M minus LSTM    | gap diff |        -0.0210 |        -0.0123 |     -0.0235 |     -0.0184 |     -0.0143 |     -0.0102 | True      | True      |

## E. H1: 80% interval quality (interval score, lower is better; coverage target 80% +/- 3)

| method                    |   coverage_old |   coverage_new |   width_old |   width_new |   interval_score_old |   interval_score_new |   interval_score_sd_old |   interval_score_sd_new | flag   |
|:--------------------------|---------------:|---------------:|------------:|------------:|---------------------:|---------------------:|------------------------:|------------------------:|:-------|
| TGT-M native              |         0.7915 |         0.7953 |      2.3193 |      2.3480 |               3.6481 |               3.6826 |                  0.0076 |                  0.0056 | MOVED  |
| TGT-M + CQR               |         0.7789 |         0.7771 |      2.2486 |      2.2472 |               3.6474 |               3.6865 |                  0.0044 |                  0.0068 | MOVED  |
| XGBoost-QR native         |         0.8059 |         0.7717 |      2.5033 |      2.3249 |               3.6977 |               3.6221 |                  0.0009 |                  0.0010 | MOVED  |
| XGBoost-QR + CQR          |         0.7850 |         0.7741 |      2.4012 |      2.3358 |               3.7000 |               3.6206 |                  0.0007 |                  0.0015 | MOVED  |
| SVR point + conformal     |         0.8029 |         0.8038 |      2.4740 |      2.4750 |               3.8831 |               3.8818 |                  0.0044 |                  0.0069 | NOISE  |
| RF point + conformal      |         0.8104 |         0.8099 |      2.4674 |      2.4626 |               3.8542 |               3.8521 |                  0.0009 |                  0.0016 | NOISE  |
| ANN point + conformal     |         0.8108 |         0.8089 |      2.4251 |      2.4169 |               3.8115 |               3.8132 |                  0.0067 |                  0.0061 | NOISE  |
| XGBoost point + conformal |         0.8103 |         0.8116 |      2.4097 |      2.4153 |               3.8367 |               3.8366 |                  0.0011 |                  0.0010 | NOISE  |
| CNN1D point + conformal   |         0.8101 |         0.8114 |      2.4034 |      2.3960 |               3.7913 |               3.7854 |                  0.0032 |                  0.0113 | NOISE  |
| LSTM point + conformal    |         0.8133 |         0.8100 |      2.4033 |      2.4003 |               3.7760 |               3.7863 |                  0.0086 |                  0.0039 | NOISE  |
| TGT-M point + conformal   |         0.8047 |         0.8063 |      2.3908 |      2.4056 |               3.8206 |               3.8321 |                  0.0078 |                  0.0091 | NOISE  |

### E2. TGT-M minus competitor, interval score (negative = TGT-M better), 95% CI

| comparison                                   |   diff_interval_score_old |   diff_interval_score_new |   ci_lo_old |   ci_hi_old |   ci_lo_new |   ci_hi_new | sig old   | sig new   | flag               |
|:---------------------------------------------|--------------------------:|--------------------------:|------------:|------------:|------------:|------------:|:----------|:----------|:-------------------|
| TGT-M native minus XGBoost-QR native         |                   -0.0496 |                    0.0605 |     -0.0647 |     -0.0342 |      0.0469 |      0.0747 | True      | False     | CHANGES CONCLUSION |
| TGT-M + CQR minus XGBoost-QR + CQR           |                   -0.0526 |                    0.0659 |     -0.0681 |     -0.0368 |      0.0518 |      0.0807 | True      | False     | CHANGES CONCLUSION |
| TGT-M native minus SVR point + conformal     |                   -0.2350 |                   -0.1992 |     -0.2514 |     -0.2197 |     -0.2154 |     -0.1844 | True      | True      | same               |
| TGT-M native minus RF point + conformal      |                   -0.2062 |                   -0.1695 |     -0.2234 |     -0.1887 |     -0.1882 |     -0.1523 | True      | True      | same               |
| TGT-M native minus ANN point + conformal     |                   -0.1634 |                   -0.1306 |     -0.1779 |     -0.1493 |     -0.1451 |     -0.1171 | True      | True      | same               |
| TGT-M native minus XGBoost point + conformal |                   -0.1886 |                   -0.1540 |     -0.2052 |     -0.1725 |     -0.1711 |     -0.1376 | True      | True      | same               |
| TGT-M native minus CNN1D point + conformal   |                   -0.1432 |                   -0.1027 |     -0.1581 |     -0.1291 |     -0.1180 |     -0.0884 | True      | True      | same               |
| TGT-M native minus LSTM point + conformal    |                   -0.1280 |                   -0.1037 |     -0.1424 |     -0.1135 |     -0.1178 |     -0.0897 | True      | True      | same               |

## F. H2: robustness to missing recent history

### F1. Degradation D = anomaly MAE(p=0.6) - MAE(p=0) and differences, 95% CI

| comparison          | what                            |   estimate_old |   estimate_new |   ci_lo_old |   ci_hi_old |   ci_lo_new |   ci_hi_new | sig old   | sig new   | flag                              |
|:--------------------|:--------------------------------|---------------:|---------------:|------------:|------------:|------------:|------------:|:----------|:----------|:----------------------------------|
| SVR                 | degradation D (p=0.6 minus p=0) |         0.0893 |         0.0868 |      0.0845 |      0.0938 |      0.0824 |      0.0914 | True      | True      | same                              |
| RF                  | degradation D (p=0.6 minus p=0) |         0.0595 |         0.0592 |      0.0550 |      0.0640 |      0.0549 |      0.0634 | True      | True      | same                              |
| ANN                 | degradation D (p=0.6 minus p=0) |         0.0874 |         0.0761 |      0.0827 |      0.0920 |      0.0718 |      0.0804 | True      | True      | same                              |
| XGBoost             | degradation D (p=0.6 minus p=0) |         0.0897 |         0.0790 |      0.0850 |      0.0942 |      0.0746 |      0.0833 | True      | True      | same                              |
| CNN1D               | degradation D (p=0.6 minus p=0) |         0.0889 |         0.0730 |      0.0841 |      0.0938 |      0.0685 |      0.0776 | True      | True      | same                              |
| LSTM                | degradation D (p=0.6 minus p=0) |         0.0911 |         0.0795 |      0.0863 |      0.0961 |      0.0751 |      0.0844 | True      | True      | same                              |
| TGT-M               | degradation D (p=0.6 minus p=0) |         0.0823 |         0.0674 |      0.0780 |      0.0865 |      0.0633 |      0.0715 | True      | True      | same                              |
| TGT-M minus SVR     | difference of degradations      |        -0.0070 |        -0.0194 |     -0.0098 |     -0.0040 |     -0.0225 |     -0.0165 | True      | True      | same                              |
| TGT-M minus SVR     | difference of MAE at p=0.6      |        -0.0347 |        -0.0475 |     -0.0379 |     -0.0312 |     -0.0512 |     -0.0437 | True      | True      | same                              |
| TGT-M minus RF      | difference of degradations      |         0.0227 |         0.0082 |      0.0193 |      0.0260 |      0.0052 |      0.0111 | True      | True      | same                              |
| TGT-M minus RF      | difference of MAE at p=0.6      |         0.0087 |        -0.0058 |      0.0051 |      0.0126 |     -0.0093 |     -0.0023 | True      | True      | same                              |
| TGT-M minus ANN     | difference of degradations      |        -0.0052 |        -0.0087 |     -0.0079 |     -0.0024 |     -0.0113 |     -0.0062 | True      | True      | same                              |
| TGT-M minus ANN     | difference of MAE at p=0.6      |        -0.0022 |        -0.0080 |     -0.0053 |      0.0010 |     -0.0109 |     -0.0052 | False     | True      | CHANGES CONCLUSION (significance) |
| TGT-M minus XGBoost | difference of degradations      |        -0.0074 |        -0.0117 |     -0.0106 |     -0.0042 |     -0.0147 |     -0.0088 | True      | True      | same                              |
| TGT-M minus XGBoost | difference of MAE at p=0.6      |        -0.0029 |        -0.0044 |     -0.0065 |      0.0007 |     -0.0078 |     -0.0012 | False     | True      | CHANGES CONCLUSION (significance) |
| TGT-M minus CNN1D   | difference of degradations      |        -0.0066 |        -0.0056 |     -0.0096 |     -0.0036 |     -0.0082 |     -0.0030 | True      | True      | same                              |
| TGT-M minus CNN1D   | difference of MAE at p=0.6      |         0.0005 |         0.0033 |     -0.0029 |      0.0039 |      0.0002 |      0.0062 | False     | True      | CHANGES CONCLUSION (significance) |
| TGT-M minus LSTM    | difference of degradations      |        -0.0088 |        -0.0121 |     -0.0118 |     -0.0058 |     -0.0149 |     -0.0093 | True      | True      | same                              |
| TGT-M minus LSTM    | difference of MAE at p=0.6      |         0.0064 |        -0.0082 |      0.0032 |      0.0095 |     -0.0116 |     -0.0050 | True      | True      | same                              |

### F2. Anomaly MAE by masking share p (mean of 3 seeds)

| model   |      p |   mean_old |   mean_new |    diff |   std_old |   std_new | flag   |
|:--------|-------:|-----------:|-----------:|--------:|----------:|----------:|:-------|
| ANN     | 0.0000 |     0.7597 |     0.7611 |  0.0014 |    0.0069 |    0.0009 | NOISE  |
| ANN     | 0.2000 |     0.7875 |     0.7854 | -0.0021 |    0.0080 |    0.0008 | NOISE  |
| ANN     | 0.4000 |     0.8169 |     0.8109 | -0.0060 |    0.0089 |    0.0026 | NOISE  |
| ANN     | 0.6000 |     0.8471 |     0.8372 | -0.0099 |    0.0106 |    0.0049 | NOISE  |
| CNN1D   | 0.0000 |     0.7556 |     0.7529 | -0.0027 |    0.0037 |    0.0043 | NOISE  |
| CNN1D   | 0.2000 |     0.7830 |     0.7751 | -0.0079 |    0.0017 |    0.0032 | MOVED  |
| CNN1D   | 0.4000 |     0.8122 |     0.7985 | -0.0137 |    0.0014 |    0.0029 | MOVED  |
| CNN1D   | 0.6000 |     0.8445 |     0.8259 | -0.0186 |    0.0042 |    0.0013 | MOVED  |
| LSTM    | 0.0000 |     0.7475 |     0.7579 |  0.0103 |    0.0020 |    0.0039 | MOVED  |
| LSTM    | 0.2000 |     0.7748 |     0.7819 |  0.0071 |    0.0014 |    0.0036 | MOVED  |
| LSTM    | 0.4000 |     0.8048 |     0.8075 |  0.0026 |    0.0021 |    0.0038 | NOISE  |
| LSTM    | 0.6000 |     0.8386 |     0.8374 | -0.0012 |    0.0039 |    0.0056 | NOISE  |
| RF      | 0.0000 |     0.7767 |     0.7758 | -0.0009 |    0.0003 |    0.0001 | MOVED  |
| RF      | 0.2000 |     0.7937 |     0.7924 | -0.0013 |    0.0004 |    0.0002 | MOVED  |
| RF      | 0.4000 |     0.8131 |     0.8120 | -0.0010 |    0.0004 |    0.0003 | MOVED  |
| RF      | 0.6000 |     0.8362 |     0.8350 | -0.0013 |    0.0006 |    0.0001 | MOVED  |
| SVR     | 0.0000 |     0.7904 |     0.7898 | -0.0006 |    0.0037 |    0.0045 | NOISE  |
| SVR     | 0.2000 |     0.8179 |     0.8148 | -0.0032 |    0.0030 |    0.0049 | NOISE  |
| SVR     | 0.4000 |     0.8479 |     0.8432 | -0.0047 |    0.0023 |    0.0055 | NOISE  |
| SVR     | 0.6000 |     0.8797 |     0.8766 | -0.0031 |    0.0031 |    0.0061 | NOISE  |
| TGT-M   | 0.0000 |     0.7627 |     0.7618 | -0.0009 |    0.0032 |    0.0025 | NOISE  |
| TGT-M   | 0.2000 |     0.7854 |     0.7804 | -0.0051 |    0.0016 |    0.0025 | MOVED  |
| TGT-M   | 0.4000 |     0.8128 |     0.8025 | -0.0103 |    0.0059 |    0.0014 | MOVED  |
| TGT-M   | 0.6000 |     0.8450 |     0.8292 | -0.0158 |    0.0143 |    0.0017 | NOISE  |
| XGBoost | 0.0000 |     0.7582 |     0.7545 | -0.0036 |    0.0009 |    0.0008 | MOVED  |
| XGBoost | 0.2000 |     0.7850 |     0.7780 | -0.0070 |    0.0014 |    0.0013 | MOVED  |
| XGBoost | 0.4000 |     0.8142 |     0.8045 | -0.0098 |    0.0017 |    0.0020 | MOVED  |
| XGBoost | 0.6000 |     0.8478 |     0.8336 | -0.0143 |    0.0020 |    0.0023 | MOVED  |

## G. Label counts and threshold sensitivity (computed from the two preprocessed tables)

### G1. Stress category counts

| category             |   pandas table |   Spark table |   diff |
|:---------------------|---------------:|--------------:|-------:|
| Normal               |         135523 |        135523 |      0 |
| Insufficient history |          52952 |         52952 |      0 |
| No reading           |          34570 |         34570 |      0 |
| Watch                |          12661 |         12661 |      0 |
| High Stress          |          10609 |         10609 |      0 |
| Moderate Stress      |           7513 |          7513 |      0 |

### G2. Readings that change category under other threshold sets

| threshold set       |   rows changed vs default (pandas) |   % of classified rows (pandas) |   moved to more severe (pandas) |   moved to less severe (pandas) |   rows changed vs default (Spark) |   % of classified rows (Spark) |   moved to more severe (Spark) |   moved to less severe (Spark) |
|:--------------------|-----------------------------------:|--------------------------------:|--------------------------------:|--------------------------------:|----------------------------------:|-------------------------------:|-------------------------------:|-------------------------------:|
| lenient (0.5/1/1.5) |                              39031 |                           23.47 |                           39031 |                               0 |                             39031 |                          23.47 |                          39031 |                              0 |
| default (1/1.5/2)   |                                  0 |                            0.00 |                               0 |                               0 |                                 0 |                           0.00 |                              0 |                              0 |
| strict (1.5/2/2.5)  |                              24413 |                           14.68 |                               0 |                           24413 |                             24413 |                          14.68 |                              0 |                          24413 |
| wide (1/2/3)        |                              14092 |                            8.47 |                               0 |                           14092 |                             14092 |                           8.47 |                              0 |                          14092 |

## H. TGT-M confusion matrices (rows = true class, columns = predicted; default thresholds, mean over seeds as saved)

**TGT-M|A, old**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |         31889 |         1777 |                    410 |                102 |
| true Watch           |          2191 |          553 |                    135 |                 30 |
| true Moderate Stress |          1064 |          362 |                    179 |                 42 |
| true High Stress     |          1147 |          348 |                    222 |                179 |

**TGT-M|A, new**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |         32731 |         1084 |                    266 |                 97 |
| true Watch           |          2403 |          382 |                     98 |                 26 |
| true Moderate Stress |          1179 |          289 |                    143 |                 36 |
| true High Stress     |          1261 |          291 |                    180 |                164 |

**TGT-M|A, new - old**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |           842 |         -693 |                   -144 |                 -5 |
| true Watch           |           212 |         -171 |                    -37 |                 -4 |
| true Moderate Stress |           115 |          -73 |                    -36 |                 -6 |
| true High Stress     |           114 |          -57 |                    -42 |                -15 |

**TGT-M|B, old**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |         32976 |          929 |                    205 |                 68 |
| true Watch           |          2381 |          382 |                    119 |                 27 |
| true Moderate Stress |          1141 |          297 |                    155 |                 54 |
| true High Stress     |          1161 |          345 |                    199 |                191 |

**TGT-M|B, new**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |         32910 |          979 |                    228 |                 61 |
| true Watch           |          2346 |          393 |                    144 |                 26 |
| true Moderate Stress |          1129 |          284 |                    172 |                 62 |
| true High Stress     |          1185 |          321 |                    207 |                183 |

**TGT-M|B, new - old**

|                      |   pred Normal |   pred Watch |   pred Moderate Stress |   pred High Stress |
|:---------------------|--------------:|-------------:|-----------------------:|-------------------:|
| true Normal          |           -66 |           50 |                     23 |                 -7 |
| true Watch           |           -35 |           11 |                     25 |                 -1 |
| true Moderate Stress |           -12 |          -13 |                     17 |                  8 |
| true High Stress     |            24 |          -24 |                      8 |                 -8 |

