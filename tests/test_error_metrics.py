"""Tests for the before/after error-metric module (hand-computed values)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gwstress import error_metrics as em  # noqa: E402


def test_metrics_match_hand_calculation():
    # y = [1, 2, 6], p = [2, 2, 8] -> e = [1, 0, 2]
    m = em.metrics([1, 2, 6], [2, 2, 8], mape_min=1.0)
    assert m["n"] == 3
    assert m["MSE"] == pytest.approx(5 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert m["MAE"] == pytest.approx(1.0)
    assert m["MedAE"] == pytest.approx(1.0)
    assert m["Bias"] == pytest.approx(1.0)                      # over-estimates
    assert m["R2"] == pytest.approx(1 - 5 / 14)                 # SS_tot = 4 + 1 + 9
    assert m["NRMSE"] == pytest.approx(np.sqrt(5 / 3) / 5)      # range = 5
    assert m["NRMSE_mean"] == pytest.approx(np.sqrt(5 / 3) / 3)  # mean = 3
    assert m["MAPE"] == pytest.approx(100 * (1 + 0 + 2 / 6) / 3)
    assert m["sMAPE"] == pytest.approx(100 * (2 / 3 + 0 + 4 / 14) / 3)


def test_mape_skips_small_actuals_and_counts_them():
    m = em.metrics([0.5, 2.0], [1.0, 3.0], mape_min=1.0)
    assert m["MAPE_n"] == 1 and m["MAPE"] == pytest.approx(50.0)
    assert m["n"] == 2                                          # other metrics still use both


def test_nan_pairs_are_dropped_and_empty_is_nan():
    m = em.metrics([1, np.nan, 3], [1, 5, np.nan])
    assert m["n"] == 1 and m["MAE"] == 0
    assert em.metrics([np.nan], [1.0])["n"] == 0


def test_perfect_forecast_has_zero_error_and_r2_one():
    m = em.metrics([2, 4, 9], [2, 4, 9])
    assert m["RMSE"] == m["MAE"] == m["Bias"] == 0 and m["R2"] == pytest.approx(1)


def test_negative_actuals_use_absolute_value_in_mape():
    m = em.metrics([-4.0, 2.0], [-2.0, 2.0], mape_min=1.0)
    assert m["MAPE"] == pytest.approx(100 * (2 / 4 + 0) / 2)


def _series():
    # 1 well, 8 years x 4 seasons: value = year index + season offset
    t = np.arange(32)
    return (t // 4 + 10 * (t % 4)).astype(float)[None, :]


def test_rules_use_only_earlier_rounds():
    v = _series()
    pred = em.predictions(v, min_prior=3)
    same, mean, prev = (pred[r] for r in em.RULES)
    assert np.isnan(same[0, :4]).all() and same[0, 4] == v[0, 0]
    assert np.isnan(prev[0, 0]) and prev[0, 5] == v[0, 4]
    # mean rule: needs 3 earlier same-season readings; first possible at year index 3
    assert np.isnan(mean[0, 4 * 2]).all() and mean[0, 4 * 3] == pytest.approx(np.mean(v[0, [0, 4, 8]]))
    assert mean[0, 4 * 4] == pytest.approx(np.mean(v[0, [0, 4, 8, 12]]))


def test_changing_future_values_never_changes_earlier_predictions():
    rng = np.random.default_rng(0)
    v = rng.uniform(1, 10, (5, 32))
    base = em.predictions(v)
    pert = v.copy()
    pert[:, 20:] = 999.0                                         # change round 20 onwards
    new = em.predictions(pert)
    for r in em.RULES:
        a, b = base[r][:, :21], new[r][:, :21]                   # predictions at rounds <= 20 only see rounds < 20
        np.testing.assert_array_equal(a[:, :20], b[:, :20])


def test_improvement_sign_convention():
    assert em.improvement(4.0, 3.0, "RMSE") == pytest.approx(25.0)      # lower is better
    assert em.improvement(0.5, 0.7, "R2") == pytest.approx(0.2)         # absolute change
    assert em.improvement(-2.0, 1.0, "Bias") == pytest.approx(50.0)     # smaller |bias|


def test_real_data_layout_and_before_after_relationships():
    pytest.importorskip("pandas")
    if not em.RAW_FILE.exists():
        pytest.skip("raw dataset not present")
    raw_same, cleaned = em.load_raw_same_wells(), em.load_cleaned()
    assert raw_same.values.shape == cleaned.values.shape == (2759, 92)
    assert (raw_same.well_ids == cleaned.well_ids).all()
    # cleaning only blanks readings (never changes a surviving value)
    both = np.isfinite(raw_same.values) & np.isfinite(cleaned.values)
    assert np.allclose(raw_same.values[both], cleaned.values[both])
    assert (np.isfinite(cleaned.values) <= np.isfinite(raw_same.values)).all()
