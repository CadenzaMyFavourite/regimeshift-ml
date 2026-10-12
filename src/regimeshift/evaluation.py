"""Aligned MAE, interval diagnostics and paired contiguous-date block uncertainty."""

from __future__ import annotations

import numpy as np
import polars as pl

from .data import KEYS
from .replay import MODELS


def score_predictions(predictions: pl.DataFrame, labels: pl.DataFrame):
    if predictions.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError("Duplicate prediction keys")
    joined = predictions.join(labels.select(KEYS + ["target"]), on=KEYS, how="left",
                              validate="1:1")
    if joined.height != labels.height or labels.join(predictions.select(KEYS), on=KEYS,
                                                     how="anti").height:
        raise ValueError("Prediction and label keys must match exactly")
    if any(joined[m].null_count() for m in MODELS) or not all(
        joined.select(pl.col(MODELS).is_finite().all()).row(0)
    ):
        raise ValueError("Every model must predict every row finitely")
    valid = joined.filter(pl.col("target").is_not_null() & pl.col("target").is_finite())
    daily_rows, interval_rows = [], []
    all_dates = joined["date_id"].unique().sort().to_list()
    # Keep empty scored dates for chronology/block sampling (zero numerator/denominator).
    for date in all_dates:
        rows = valid.filter(pl.col("date_id") == date)
        regime = int(joined.filter(pl.col("date_id") == date)["regime_id"][0])
        y = rows["target"].to_numpy()
        for model in MODELS:
            error = np.abs(rows[model].to_numpy() - y)
            daily_rows.append({"date_id": date, "regime_id": regime, "model": model,
                               "count": len(y), "absolute_error_sum": float(error.sum()),
                               "mae": float(error.mean()) if len(y) else None})
            for kind in ["fixed", "rolling"]:
                radius = rows[f"{model}_{kind}_radius"].to_numpy()
                eligible = np.isfinite(radius)
                n = int(eligible.sum())
                interval_rows.append({
                    "date_id": date, "regime_id": regime, "model": model, "interval": kind,
                    "count": n, "covered": int((error[eligible] <= radius[eligible]).sum()),
                    "width_sum": float((2 * radius[eligible]).sum()),
                })
    daily = pl.DataFrame(daily_rows)
    intervals = pl.DataFrame(interval_rows)
    overall = aggregate_errors(daily, ["model"])
    by_regime = aggregate_errors(daily, ["regime_id", "model"])
    return {
        "daily": daily, "overall": overall, "by_regime": by_regime,
        "interval_daily": intervals,
        "interval_overall": aggregate_intervals(intervals, ["model", "interval"]),
        "interval_by_regime": aggregate_intervals(intervals, ["regime_id", "model", "interval"]),
    }


def aggregate_errors(daily, groups):
    return daily.group_by(groups).agg(
        pl.col("count").sum(), pl.col("absolute_error_sum").sum(),
        (pl.col("count") > 0).sum().alias("scored_dates"),
    ).with_columns(
        pl.when(pl.col("count") > 0).then(pl.col("absolute_error_sum") / pl.col("count"))
        .otherwise(None).alias("mae")
    ).sort(groups)


def aggregate_intervals(frame, groups):
    return frame.group_by(groups).agg(pl.col("count").sum(), pl.col("covered").sum(),
                                    pl.col("width_sum").sum()).with_columns(
        pl.when(pl.col("count") > 0).then(pl.col("covered") / pl.col("count"))
        .otherwise(None).alias("coverage"),
        pl.when(pl.col("count") > 0).then(pl.col("width_sum") / pl.col("count"))
        .otherwise(None).alias("mean_width"),
    ).sort(groups)


def paired_bootstrap(daily, reference, block=5, reps=2000, seed=42):
    left = daily.filter(pl.col("model") == "regime").sort("date_id")
    right = daily.filter(pl.col("model") == reference).sort("date_id")
    if not left.select("date_id", "count").equals(right.select("date_id", "count")):
        raise ValueError("Paired models require identical dates and row counts")
    n = left.height
    if n < block or block < 1 or reps < 1:
        raise ValueError("Insufficient dates or invalid bootstrap settings")
    counts = left["count"].to_numpy()
    deltas = left["absolute_error_sum"].to_numpy() - right["absolute_error_sum"].to_numpy()
    if counts.sum() == 0:
        raise ValueError("No scored rows")
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(reps):
        starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
        indices = (starts[:, None] + np.arange(block)).ravel()[:n]
        denominator = counts[indices].sum()
        if denominator:
            estimates.append(float(deltas[indices].sum() / denominator))
    if len(estimates) < reps * .9:
        raise ValueError("Too few nonempty bootstrap samples; inspect missing target dates")
    low, high = np.quantile(estimates, [.025, .975])
    return {"model": "regime", "reference": reference,
            "mae_difference": float(deltas.sum() / counts.sum()),
            "ci_low": float(low), "ci_high": float(high), "block_dates": block,
            "replicates": len(estimates), "requested_replicates": reps,
            "interpretation": "negative favors regime; conditional on this realized replay path"}
