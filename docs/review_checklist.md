# Course review checklist

This maps each review item to the file(s) that deliver it. Paths are relative to the repository root.

| # | Review item | Owner | Delivered by | Status |
|---|---|---|---|---|
| 1 | Dataset | me | `docs/review_preprocessing_section.md` § Dataset (description table + CHIRPS paragraph); `reports/stage1_inspection.md`; `docs/preprocessing_methods.md` | done |
| 2 | Proposed Model | **owner: teammate (modeling)** | (input contract: `docs/handoff_for_modeling.md`) | to do (teammate) |
| 3 | Overall System Architecture Diagram | me (preprocessing part); teammates extend | `docs/architecture_preprocessing.png` (300 dpi), source `docs/architecture_preprocessing.mmd`, explanation `docs/architecture.md`. Modeling and dashboard boxes are dashed placeholders | preprocessing part done |
| 4 | Preprocessing (Big Data Analysis and its steps) | me | `docs/review_preprocessing_section.md` § Preprocessing: Big Data Analysis and its steps; detail in `docs/preprocessing_methods.md`; code in `src/gwstress/`, `scripts/` | done |
| 5 | Results of Preprocessing | me | `reports/figures/fig01`–`fig16` (17 PNGs, 300 dpi), `reports/tables/` (CSV + Markdown), `reports/figure_captions.md`; § Results of preprocessing in the review section; `reports/data_quality.md`, `reports/data_validity.md` | done |
| 6 | Classification | **owner: teammate (modeling)** | inputs: `processed/gw_features_by_state/`, `processed/splits.parquet`, labels `stress_category` (primary), `stress_tadj`, `stress_roll10` | to do (teammate) |
| 7 | Results of Classification | **owner: teammate (modeling)** | (use `eval_eligible`; report the extension separately with per-round counts) | to do (teammate) |
| 8 | Comparison with existing classifiers | **owner: teammate (modeling)** | none | to do (teammate) |
| 9 | Plan of Action / Member Tasks | **owner: me, manual** | none (write by hand) | to do (manual) |
| 10 | References | me | `docs/references.md` (IEEE). **All 15 entries are marked [verify]**: check them before submission | done, needs verification |
| 11 | Geotagged photo from Review 2 | **owner: me, manual** | none (add the photo by hand) | to do (manual) |
| 12 | Colab link | me | badge in `notebooks/preprocessing_colab.ipynb` and `notebooks/README.md` → https://colab.research.google.com/github/dhruvvvgg/bde-team7/blob/main/notebooks/preprocessing_colab.ipynb ; executed copy `notebooks/preprocessing_colab_executed.ipynb` | done (tested with nbconvert, not in real Colab; the repo must be public for the link to work without a token) |

## Before submission

- **References:** open each DOI in `docs/references.md`, correct anything wrong, and remove the
  `[verify]` tags.
- **Colab link:** open the badge link while logged out, to confirm the repository is public and the
  notebook loads. Then run it once in real Colab.
- **Teammates' stages:** add the modeling and dashboard nodes to `docs/architecture_preprocessing.mmd`
  (or `scripts/13_architecture.py`).
- **Manual items:** add the items 9 and 11 material.
