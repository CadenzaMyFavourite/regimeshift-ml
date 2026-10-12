"""Development and final evaluation are separate, artifact-verified commands."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits

from .config import Config
from .data import KEYS, load_dates, split_dates, verify_dataset
from .evaluation import paired_bootstrap, score_predictions
from .features import build_features, daily_summaries
from .models import fit_experts
from .regimes import RegimeModel
from .replay import Outcome, ReplayEngine, prediction_frame, simulate_cached
from .utils import file_hash, provenance, read_json, write_json


def cache_predictions(features, experts, regimes, previous_summary):
    predictions = experts.predict(features)
    result = features.select(KEYS + ["target"]).with_columns(
        pl.Series("ridge", predictions[:, 0]), pl.Series("lightgbm", predictions[:, 1])
    )
    labels = []
    for summary in daily_summaries(features).iter_rows(named=True):
        labels.append({"date_id": summary["date_id"], "regime_id": regimes.assign(previous_summary)})
        previous_summary = summary
    return result.join(pl.DataFrame(labels).cast({"date_id": pl.Int32}), on="date_id")\
        .sort(KEYS), previous_summary


def choose_adaptation(cache, scale, fixed_weight, config):
    records = []
    for half_life in config.half_lives:
        for temperature in config.temperatures:
            params = {"half_life": half_life, "temperature": temperature}
            result = simulate_cached(cache, params, params, fixed_weight, scale,
                                     config.shrinkage_dates)
            records.append({**params, "global_mae": result["global"],
                            "regime_mae": result["regime"]})
    chosen = {}
    for name in ["global", "regime"]:
        best = min(records, key=lambda row: row[f"{name}_mae"])
        chosen[name] = {k: best[k] for k in ["half_life", "temperature"]}
    return chosen, records


def _persist_day(engine, day, date, phase, next_date, output, events):
    regime = int(day["regime_id"][0])
    experts = day.select(["ridge", "lightgbm"]).to_numpy()
    predictions, event = engine.predict(date, regime, experts)
    result = prediction_frame(day, predictions, event, next_date, phase)
    result.write_parquet(output / f"{date}.parquet", compression="zstd")
    # Label values are deliberately read only after the prediction artifact exists.
    outcome = Outcome(date, regime, experts, predictions, day["target"].to_numpy(), phase)
    engine.queue_labels(outcome)
    events.write(json.dumps({**event, "phase": phase}, allow_nan=False) + "\n")
    return result, outcome


def develop(config: Config) -> Path:
    config.validate()
    start = time.perf_counter()
    dataset, run = Path(config.dataset).resolve(), Path(config.run_dir).resolve()
    config.dataset, config.run_dir = str(dataset), str(run)
    if run.exists():
        raise FileExistsError(f"Run exists; choose a new run_dir: {run}")
    manifest = verify_dataset(dataset)
    splits = split_dates(manifest["dates"])
    if len(splits["test"]) < config.bootstrap_block:
        raise ValueError("Final evaluation must contain at least one bootstrap block")
    run.mkdir(parents=True)
    write_json(run / "config.json", config.to_dict())
    write_json(run / "splits.json", splits)  # persisted before any fit
    write_json(run / "dataset.json", manifest)
    write_json(run / "environment.json", provenance())
    train = build_features(load_dates(dataset, splits["train"]))
    tune = build_features(load_dates(dataset, splits["tune"]))
    train_summaries = daily_summaries(train)
    train_summaries.write_parquet(run / "training_summaries.parquet")
    print(f"Fitting experts on {train.height:,} rows across {len(splits['train'])} dates.",
          flush=True)
    fit_start = time.perf_counter()
    with threadpool_limits(limits=config.threads):
        experts, model_scores, scale = fit_experts(train, tune, config)
        regimes = RegimeModel.fit(train_summaries, config.seed)
        cache, previous_summary = cache_predictions(
            tune, experts, regimes, train_summaries.row(-1, named=True)
        )
    fit_seconds = time.perf_counter() - fit_start
    del train, tune
    cache.write_parquet(run / "tuning_cache.parquet")
    write_json(run / "regime_centroids.json", regimes.describe())
    y = cache["target"].to_numpy()
    valid = np.isfinite(y)
    expert_predictions = cache.select(["ridge", "lightgbm"]).to_numpy()
    fixed_grid = [{"ridge_weight": float(w),
                   "tune_mae": float(np.abs(y[valid] - expert_predictions[valid] @ [w, 1-w]).mean())}
                  for w in np.linspace(0, 1, 21)]
    fixed_weight = min(fixed_grid, key=lambda row: row["tune_mae"])["ridge_weight"]
    chosen, adaptation_scores = choose_adaptation(cache, scale, fixed_weight, config)
    tune_scores = simulate_cached(cache, chosen["global"], chosen["regime"], fixed_weight,
                                  scale, config.shrinkage_dates)
    reference = min(["zero", "ridge", "lightgbm", "fixed"], key=lambda m: tune_scores[m])
    selection = {
        "global_params": chosen["global"], "regime_params": chosen["regime"],
        "fixed_ridge_weight": fixed_weight, "loss_scale": scale, "static_reference": reference,
        "model_candidates": model_scores, "blend_candidates": fixed_grid,
        "adaptation_candidates": adaptation_scores, "tuning_scores": tune_scores,
        "note": "Tuning scores select settings; they are not unbiased performance estimates.",
    }
    write_json(run / "selection.json", selection)
    engine = ReplayEngine(chosen["global"], chosen["regime"], fixed_weight, scale,
                          config.shrinkage_dates, config.interval_coverage, config.interval_window)
    output = run / "development_predictions"
    output.mkdir()
    all_dates = manifest["dates"]
    next_dates = dict(zip(all_dates[:-1], all_dates[1:]))
    residuals = []
    print("Replaying tuning and calibration dates; final test labels remain unused.", flush=True)
    with (run / "development_events.jsonl").open("w") as events:
        for (date,), day in cache.partition_by("date_id", as_dict=True, maintain_order=True).items():
            _persist_day(engine, day, date, "tune", next_dates[date], output, events)
        for date in splits["calibration"]:
            features = build_features(load_dates(dataset, [date]))
            with threadpool_limits(limits=config.threads):
                day, previous_summary = cache_predictions(features, experts, regimes, previous_summary)
            _, outcome = _persist_day(engine, day, date, "calibration", next_dates[date], output, events)
            residuals.append(outcome.residuals())
    # These constants are used only in TEST, after every calibration target is eligible.
    engine.intervals.freeze(residuals)
    joblib.dump({"experts": experts, "regimes": regimes}, run / "models.joblib")
    joblib.dump({"engine": engine, "previous_summary": previous_summary}, run / "checkpoint.joblib")
    write_json(run / "development_runtime.json", {
        "fit_and_tuning_prediction_seconds": fit_seconds,
        "total_seconds": time.perf_counter() - start,
    })
    frozen_names = ["config.json", "splits.json", "selection.json", "models.joblib",
                    "checkpoint.joblib", "dataset.json", "environment.json"]
    write_json(run / "frozen.json", {
        "status": "ready_for_final_evaluation", "source_kind": manifest["source_kind"],
        "dataset_manifest_hash": file_hash(dataset / "manifest.json"),
        "artifacts": {name: file_hash(run / name) for name in frozen_names},
    })
    return run


def verify_run(run):
    frozen = read_json(run / "frozen.json")
    for name, digest in frozen["artifacts"].items():
        if file_hash(run / name) != digest:
            raise ValueError(f"Frozen development artifact changed: {name}")
    return frozen


def verify_execution_environment(run):
    """Do not silently replay a developed study under changed scientific code or libraries."""
    expected, current = read_json(run / "environment.json"), provenance()
    scientific_modules = ["adaptation.py", "config.py", "data.py", "evaluation.py", "features.py",
                          "models.py", "pipeline.py", "regimes.py", "replay.py"]
    for name in scientific_modules:
        if expected["source_hashes"][name] != current["source_hashes"][name]:
            raise ValueError(f"Scientific code changed after development: {name}; develop a new run")
    if expected["packages"] != current["packages"] or expected["python"] != current["python"]:
        raise ValueError("Python or scientific dependencies changed; restore the recorded environment")


def evaluate(run: Path) -> Path:
    start = time.perf_counter()
    run = run.resolve()
    frozen = verify_run(run)
    config = Config.from_dict(read_json(run / "config.json"))
    dataset = Path(config.dataset)
    manifest = verify_dataset(dataset, frozen["dataset_manifest_hash"])
    splits = read_json(run / "splits.json")
    selection = read_json(run / "selection.json")
    output = run / "evaluation"
    if output.exists():
        raise FileExistsError("Evaluation already exists (possibly incomplete); use a new run")
    verify_execution_environment(run)
    # Only load local, hash-verified artifacts produced by this application.
    models = joblib.load(run / "models.joblib")
    checkpoint = joblib.load(run / "checkpoint.joblib")
    engine, previous_summary = checkpoint["engine"], checkpoint["previous_summary"]
    output.mkdir()
    predictions_dir = output / "predictions"
    predictions_dir.mkdir()
    results, runtime_rows = [], []
    next_dates = dict(zip(manifest["dates"][:-1], manifest["dates"][1:]))
    print(f"Evaluating {len(splits['test'])} future dates with frozen choices.", flush=True)
    with (output / "events.jsonl").open("w") as events:
        for date in splits["test"]:
            day_start = time.perf_counter()
            features = build_features(load_dates(dataset, [date]))
            with threadpool_limits(limits=config.threads):
                day, previous_summary = cache_predictions(
                    features, models["experts"], models["regimes"], previous_summary
                )
            pred, _ = _persist_day(engine, day, date, "test", next_dates.get(date),
                                   predictions_dir, events)
            # Scoring is outside the prediction interface and never feeds model selection.
            results.append(score_predictions(pred, day))
            runtime_rows.append({"date_id": date, "rows": day.height,
                                 "seconds": time.perf_counter() - day_start})
    from .evaluation import aggregate_errors, aggregate_intervals

    daily = pl.concat([r["daily"] for r in results])
    interval_daily = pl.concat([r["interval_daily"] for r in results])
    if daily["count"].sum() == 0:
        raise ValueError("Final evaluation has no finite targets")
    tables = {
        "daily_metrics": daily, "metrics": aggregate_errors(daily, ["model"]),
        "regime_metrics": aggregate_errors(daily, ["regime_id", "model"]),
        "interval_daily": interval_daily,
        "interval_metrics": aggregate_intervals(interval_daily, ["model", "interval"]),
        "interval_regime_metrics": aggregate_intervals(interval_daily, ["regime_id", "model", "interval"]),
        "runtime": pl.DataFrame(runtime_rows),
    }
    for name, table in tables.items():
        table.write_parquet(output / f"{name}.parquet")
        table.write_csv(output / f"{name}.csv")
    comparisons = [paired_bootstrap(daily, reference, config.bootstrap_block,
                                     config.bootstrap_reps, config.seed)
                   for reference in [selection["static_reference"], "global"]]
    write_json(output / "comparisons.json", comparisons)
    joblib.dump({"engine": engine, "previous_summary": previous_summary}, output / "final_state.joblib")
    write_json(output / "summary.json", {
        "status": "complete", "source_kind": frozen["source_kind"],
        "target_units": manifest["target_units"], "static_reference": selection["static_reference"],
        "test_dates": splits["test"], "metrics": tables["metrics"].to_dicts(),
        "comparisons": comparisons, "total_seconds": time.perf_counter() - start,
        "coverage_target": config.interval_coverage, "evaluation_environment": provenance(),
        "limitations": ["Conditional uncertainty on one realized replay path, not repeated training.",
                        "Previous-date labels and summaries are an offline timing assumption.",
                        "No guaranteed interval coverage under dependence or distribution shift.",
                        "Frozen experts cannot learn new feature relationships."],
    })
    return output
