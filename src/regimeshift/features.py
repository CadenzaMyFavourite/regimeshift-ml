"""Features depend on the current row and earlier rows of the same stock/date."""

import polars as pl

from .data import KEYS, RAW_FEATURES

DERIVED = ["spread_bps", "size_imbalance", "signed_auction_imbalance", "auction_ratio",
           "log_liquidity", "log_matched", "wap_reference_bps", "mid_reference_bps",
           "near_reference_bps", "far_reference_bps", "wap_return_10", "wap_return_30",
           "wap_return_60", "wap_std_6obs", "spread_mean_6obs"]
NUMERIC_FEATURES = ["seconds_in_bucket"] + RAW_FEATURES + DERIVED


def ratio(a, b):
    return pl.when(b.abs() > 1e-12).then(a / b).otherwise(None)


def build_features(frame: pl.DataFrame) -> pl.DataFrame:
    group = ["date_id", "stock_id"]
    f = frame.sort(["date_id", "stock_id", "seconds_in_bucket"])
    c = pl.col
    f = f.with_columns(
        (1e4 * ratio(c("ask_price") - c("bid_price"), c("reference_price"))).alias("spread_bps"),
        ratio(c("bid_size") - c("ask_size"), c("bid_size") + c("ask_size")).alias("size_imbalance"),
        (c("imbalance_size") * c("imbalance_buy_sell_flag")).alias("signed_auction_imbalance"),
        ratio(c("imbalance_size") * c("imbalance_buy_sell_flag"),
              c("matched_size") + c("imbalance_size")).alias("auction_ratio"),
        (c("bid_size") + c("ask_size")).log1p().alias("log_liquidity"),
        c("matched_size").log1p().alias("log_matched"),
        (1e4 * (ratio(c("wap"), c("reference_price")) - 1)).alias("wap_reference_bps"),
        (1e4 * (ratio((c("bid_price") + c("ask_price")) / 2, c("reference_price")) - 1))
        .alias("mid_reference_bps"),
        *[(1e4 * (ratio(c(f"{name}_price"), c("reference_price")) - 1))
          .alias(f"{name}_reference_bps") for name in ["near", "far"]],
    )
    # Exact time joins avoid pretending that six irregular observations equal 60 seconds.
    for seconds in [10, 30, 60]:
        prior = f.select(group + ["seconds_in_bucket", "wap"]).with_columns(
            (c("seconds_in_bucket") + seconds).alias("seconds_in_bucket")
        ).rename({"wap": "prior_wap"})
        f = f.join(prior, on=group + ["seconds_in_bucket"], how="left").with_columns(
            (1e4 * (ratio(c("wap"), c("prior_wap")) - 1)).alias(f"wap_return_{seconds}")
        ).drop("prior_wap")
    f = f.sort(group + ["seconds_in_bucket"]).with_columns(
        (1e4 * c("wap").rolling_std(window_size=6, min_samples=2).over(group))
        .alias("wap_std_6obs"),
        c("spread_bps").rolling_mean(window_size=6, min_samples=1).over(group)
        .alias("spread_mean_6obs"),
    )
    f = f.with_columns([
        pl.when(c(name).is_finite()).then(c(name)).otherwise(None).cast(pl.Float32).alias(name)
        for name in NUMERIC_FEATURES if name not in KEYS
    ])
    return f.sort(KEYS)


SUMMARY_FEATURES = ["spread", "imbalance", "liquidity", "wap_variability"]


def daily_summaries(features: pl.DataFrame) -> pl.DataFrame:
    return features.group_by("date_id").agg(
        pl.col("spread_bps").median().alias("spread"),
        pl.col("size_imbalance").abs().median().alias("imbalance"),
        pl.col("log_liquidity").median().alias("liquidity"),
        pl.col("wap_return_10").std().alias("wap_variability"),
    ).sort("date_id")
