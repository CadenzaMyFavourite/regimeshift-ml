"""Local dataset import, integrity checks and whole-date splits."""

from __future__ import annotations

import uuid
from pathlib import Path

import polars as pl

from .utils import file_hash, read_json, write_json

KEYS = ["date_id", "seconds_in_bucket", "stock_id"]
RAW_FEATURES = [
    "imbalance_size", "imbalance_buy_sell_flag", "reference_price", "matched_size",
    "far_price", "near_price", "bid_price", "bid_size", "ask_price", "ask_size", "wap",
]
REQUIRED = KEYS + RAW_FEATURES + ["target"]


def validate_frame(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict]:
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not frame.height:
        raise ValueError("Dataset is empty")
    frame = frame.select(REQUIRED).with_columns(
        pl.all().cast(pl.Float64, strict=True).fill_nan(None)
    )
    infinite = frame.select(pl.all().is_infinite().sum()).row(0, named=True)
    if any(infinite.values()):
        raise ValueError(f"Infinite numeric values: {infinite}")
    for key in KEYS:
        series = frame[key]
        if series.null_count() or (series != series.floor()).any() or (series < 0).any():
            raise ValueError(f"{key} must contain nonnegative integers without missing values")
    if frame.filter((pl.col("seconds_in_bucket") > 540)
                    | (pl.col("seconds_in_bucket") % 10 != 0)).height:
        raise ValueError("seconds_in_bucket must be 0..540 in ten-second increments")
    if frame.select(pl.struct(KEYS).is_duplicated().any()).item():
        raise ValueError("Duplicate stock/date/time keys")
    for name in ["imbalance_size", "matched_size", "bid_size", "ask_size"]:
        if (frame[name] < 0).any():
            raise ValueError(f"Negative size in {name}")
    for name in ["wap", "reference_price", "bid_price", "ask_price"]:
        if (frame[name] <= 0).any():
            raise ValueError(f"Nonpositive price in {name}")
    if frame.filter(~pl.col("imbalance_buy_sell_flag").is_in([-1, 0, 1])).height:
        raise ValueError("imbalance_buy_sell_flag must be -1, 0, 1 or missing")
    if frame.filter(pl.col("ask_price") < pl.col("bid_price")).height:
        raise ValueError("Crossed bid/ask prices")
    frame = frame.with_columns(pl.col(KEYS).cast(pl.Int32))
    ordered = frame.sort(KEYS)
    audit = {
        "rows": frame.height, "stocks": frame["stock_id"].n_unique(),
        "input_was_sorted": frame.select(KEYS).equals(ordered.select(KEYS)),
        "null_counts": frame.null_count().row(0, named=True),
        "valid_targets": frame["target"].count(),
        "date_counts": ordered.group_by("date_id", maintain_order=True).agg(
            pl.len().alias("rows"), pl.col("target").count().alias("valid_targets")
        ).to_dicts(),
    }
    return ordered, audit


def prepare(source: Path, destination: Path, source_kind: str = "optiver") -> Path:
    if source_kind not in {"synthetic", "optiver"}:
        raise ValueError("source_kind must be synthetic or optiver")
    if destination.exists():
        raise FileExistsError(f"Destination exists; choose a new directory: {destination}")
    # Only required columns are materialized. The Optiver-sized raw frame fits in 32 GiB;
    # downstream fitting and evaluation load only their selected date partitions.
    schema = {name: pl.Float64 for name in REQUIRED}
    frame = pl.read_csv(source, schema_overrides=schema, null_values=["", "NaN", "nan"])
    frame, audit = validate_frame(frame)
    staging = destination.with_name(destination.name + ".preparing-" + uuid.uuid4().hex[:8])
    staging.mkdir(parents=True)
    partitions = {}
    dates = frame["date_id"].unique().sort().to_list()
    for (date,), daily in frame.partition_by("date_id", as_dict=True, maintain_order=True).items():
        name = f"date_id={date}/part.parquet"
        path = staging / name
        path.parent.mkdir()
        daily.write_parquet(path, compression="zstd")
        partitions[str(date)] = {"file": name, "sha256": file_hash(path)}
    manifest = {
        "format_version": 1, "source_kind": source_kind,
        "source_file": str(source.resolve()), "source_sha256": file_hash(source),
        "target_units": "synthetic basis-point-like units" if source_kind == "synthetic"
                        else "basis points; provided Optiver index-relative target",
        "dates": dates, "partitions": partitions, "audit": audit,
    }
    write_json(staging / "manifest.json", manifest)
    staging.rename(destination)
    return destination


def verify_dataset(path: Path, expected_hash: str | None = None) -> dict:
    manifest_path = path / "manifest.json"
    if expected_hash and file_hash(manifest_path) != expected_hash:
        raise ValueError("Dataset manifest changed after development")
    manifest = read_json(manifest_path)
    for part in manifest["partitions"].values():
        if file_hash(path / part["file"]) != part["sha256"]:
            raise ValueError(f"Dataset partition changed: {part['file']}")
    return manifest


def load_dates(path: Path, dates: list[int]) -> pl.DataFrame:
    if not dates:
        raise ValueError("Cannot load an empty date block")
    return pl.concat([pl.read_parquet(path / f"date_id={d}" / "part.parquet")
                      for d in dates]).sort(KEYS)


def split_dates(dates: list[int]) -> dict[str, list[int]]:
    dates = sorted(set(dates))
    if len(dates) < 50:
        raise ValueError("At least 50 dates are required for four blocks and block uncertainty")
    n = len(dates)
    a, b, c = int(n * .6), int(n * .7), int(n * .8)
    return {"train": dates[:a], "tune": dates[a:b], "calibration": dates[b:c],
            "test": dates[c:]}
