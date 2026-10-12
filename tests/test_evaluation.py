import numpy as np
import polars as pl
import pytest

from regimeshift.evaluation import paired_bootstrap, score_predictions
from regimeshift.replay import MODELS, prediction_frame


def test_hand_calculated_metrics_and_alignment():
    keys = pl.DataFrame({"date_id": [0, 0, 0], "stock_id": [0, 1, 2], "seconds_in_bucket": [0, 0, 0]})
    predictions = {m: np.array([1., 2., 3.]) for m in MODELS}
    event = {"regime": 0, "global_ridge_weight": .5, "regime_ridge_weight": .5,
             "fixed_radii": {m: 1. for m in MODELS}, "rolling_radii": {m: 2. for m in MODELS}}
    frame = prediction_frame(keys, predictions, event, 1, "test")
    labels = keys.with_columns(pl.Series("target", [0., 4., 3.])).reverse()
    result = score_predictions(frame, labels)
    assert result["overall"]["mae"].to_list() == [1.] * 6
    fixed = result["interval_overall"].filter(pl.col("interval") == "fixed")
    assert fixed["coverage"].to_list() == pytest.approx([2/3] * 6)
    assert fixed["mean_width"].to_list() == [2.] * 6
    with pytest.raises(ValueError, match="keys must match"):
        score_predictions(frame, labels.with_columns((pl.col("stock_id") + 10).alias("stock_id")))


def test_block_bootstrap_preserves_row_weighted_delta():
    rows = []
    for d, n in enumerate([1, 10, 2, 30, 4, 20]):
        rows += [{"date_id": d, "count": n, "model": model, "absolute_error_sum": n * error}
                 for model, error in [("regime", 1.), ("fixed", 2.)]]
    result = paired_bootstrap(pl.DataFrame(rows), "fixed", block=2, reps=100)
    assert result["mae_difference"] == -1.
    assert result["ci_low"] == -1. and result["ci_high"] == -1.


def test_empty_scored_date_is_retained():
    keys = pl.DataFrame({"date_id": [5], "stock_id": [0], "seconds_in_bucket": [0]})
    event = {"regime": 0, "global_ridge_weight": .5, "regime_ridge_weight": .5,
             "fixed_radii": {m: 1. for m in MODELS}, "rolling_radii": {}}
    predictions = prediction_frame(keys, {m: np.array([0.]) for m in MODELS}, event, 6, "test")
    result = score_predictions(predictions, keys.with_columns(pl.lit(None, dtype=pl.Float64).alias("target")))
    assert result["daily"]["date_id"].to_list() == [5] * 6
    assert result["daily"]["count"].sum() == 0
    assert result["overall"]["mae"].null_count() == 6


def test_bootstrap_is_seeded_and_rejects_mismatched_counts():
    rows = []
    for d in range(10):
        rows += [{"date_id": d, "count": d+1, "model": model, "absolute_error_sum": float(error)}
                 for model, error in [("regime", d*d), ("fixed", d+1)]]
    frame = pl.DataFrame(rows)
    assert paired_bootstrap(frame, "fixed") == paired_bootstrap(frame, "fixed")
    mismatch = frame.with_columns(pl.when(pl.col("model") == "fixed").then(pl.col("count")+1)
                                  .otherwise(pl.col("count")).alias("count"))
    with pytest.raises(ValueError, match="identical dates"):
        paired_bootstrap(mismatch, "fixed")
