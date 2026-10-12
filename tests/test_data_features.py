import numpy as np
import polars as pl
import pytest

from regimeshift.data import KEYS, prepare, split_dates, validate_frame, verify_dataset
from regimeshift.features import NUMERIC_FEATURES, build_features


def test_whole_date_splits():
    splits = split_dates(list(range(100)))
    assert [len(x) for x in splits.values()] == [60, 10, 10, 20]
    assert max(splits["train"]) < min(splits["tune"])
    assert sum(splits.values(), []) == list(range(100))
    with pytest.raises(ValueError, match="50 dates"):
        split_dates(list(range(20)))


def test_import_sorts_and_preserves_missing_targets(raw_frame):
    raw = raw_frame.reverse()
    clean, audit = validate_frame(raw)
    assert not audit["input_was_sorted"]
    assert clean.select(KEYS).equals(raw_frame.select(KEYS))
    assert clean["target"].null_count() == raw_frame["target"].null_count()


@pytest.mark.parametrize("mutation,match", [
    (lambda f: pl.concat([f, f.head(1)]), "Duplicate"),
    (lambda f: f.drop("wap"), "Missing required"),
    (lambda f: f.with_columns(pl.lit(float("inf")).alias("bid_size")), "Infinite"),
    (lambda f: f.with_columns(pl.lit(-1).alias("bid_size")), "Negative"),
    (lambda f: f.with_columns(pl.lit(.5).alias("stock_id")), "integers"),
    (lambda f: f.with_columns(pl.lit(None).alias("date_id")), "integers"),
    (lambda f: f.with_columns(pl.lit(7).alias("seconds_in_bucket")), "ten-second"),
    (lambda f: f.with_columns(pl.lit(4).alias("imbalance_buy_sell_flag")), "flag"),
    (lambda f: f.with_columns((pl.col("ask_price") + 1).alias("bid_price")), "Crossed"),
])
def test_bad_inputs_are_rejected(raw_frame, mutation, match):
    with pytest.raises(ValueError, match=match):
        validate_frame(mutation(raw_frame))


def test_future_features_cannot_change_past(raw_frame):
    before = build_features(raw_frame)
    changed = raw_frame.with_columns(
        pl.when((pl.col("date_id") > 10) | ((pl.col("date_id") == 10)
                & (pl.col("seconds_in_bucket") > 30)))
        .then(pl.col("wap") * 2).otherwise(pl.col("wap")).alias("wap")
    )
    after = build_features(changed)
    mask = (pl.col("date_id") < 10) | ((pl.col("date_id") == 10)
                                         & (pl.col("seconds_in_bucket") <= 30))
    assert before.filter(mask).equals(after.filter(mask))


def test_target_changes_cannot_change_features(raw_frame):
    first = build_features(raw_frame).select(NUMERIC_FEATURES)
    second = build_features(raw_frame.with_columns(pl.lit(1e9).alias("target")))
    assert first.equals(second.select(NUMERIC_FEATURES))


def test_lags_reset_and_respect_exact_seconds(raw_frame):
    features = build_features(raw_frame)
    assert features.select(KEYS).schema == raw_frame.select(KEYS).schema
    starts = features.filter(pl.col("seconds_in_bucket") == 0)
    for name in ["wap_return_10", "wap_return_30", "wap_return_60", "wap_std_6obs"]:
        assert starts[name].null_count() == starts.height
    missing = raw_frame.filter(pl.col("seconds_in_bucket") != 10)
    result = build_features(missing).filter(pl.col("seconds_in_bucket") == 20)
    assert result["wap_return_10"].null_count() == result.height
    example = raw_frame.filter((pl.col("date_id") == 0) & (pl.col("stock_id") == 0)).sort("seconds_in_bucket")
    feature = features.filter((pl.col("date_id") == 0) & (pl.col("stock_id") == 0)
                              & (pl.col("seconds_in_bucket") == 10))["wap_return_10"][0]
    assert feature == pytest.approx((example["wap"][1] / example["wap"][0] - 1) * 1e4, rel=1e-5)


def test_zero_size_ratios_are_missing_not_infinite(raw_frame):
    zero = raw_frame.with_columns(pl.lit(0).alias("bid_size"), pl.lit(0).alias("ask_size"))
    features = build_features(zero)
    assert features["size_imbalance"].null_count() == features.height
    assert not np.isinf(features.select(NUMERIC_FEATURES).to_numpy()).any()


def test_dataset_hash_detects_changed_partition(raw_frame, tmp_path):
    csv = tmp_path / "raw.csv"
    raw_frame.write_csv(csv)
    destination = prepare(csv, tmp_path / "processed", "synthetic")
    verify_dataset(destination)
    part = destination / "date_id=0/part.parquet"
    pl.read_parquet(part).with_columns(pl.lit(3.).alias("target")).write_parquet(part)
    with pytest.raises(ValueError, match="partition changed"):
        verify_dataset(destination)
