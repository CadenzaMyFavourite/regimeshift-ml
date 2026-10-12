import json

import joblib
import numpy as np
import polars as pl
import pytest

from regimeshift.config import Config
from regimeshift.data import load_dates, prepare
from regimeshift.features import build_features
from regimeshift.pipeline import (
    cache_predictions,
    develop,
    evaluate,
    verify_execution_environment,
    verify_run,
)
from regimeshift.reporting import report
from regimeshift.utils import file_hash, read_json


@pytest.fixture(scope="module")
def study(raw_frame, tmp_path_factory):
    root = tmp_path_factory.mktemp("study")
    csv = root / "synthetic.csv"
    raw_frame.write_csv(csv)
    dataset = prepare(csv, root / "processed", "synthetic")
    config = Config(dataset=str(dataset), run_dir=str(root / "run"), threads=2,
                    ridge_alphas=(10.,), lgbm_leaves=(7,), lgbm_estimators=20,
                    lgbm_min_child_samples=10, half_lives=(5, 10), temperatures=(.5,),
                    bootstrap_reps=100)
    # Assert that development never loads test dates for modeling or selection.
    import regimeshift.pipeline as pipeline
    original = pipeline.load_dates
    observed = []

    def track(path, dates):
        observed.extend(dates)
        return original(path, dates)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pipeline, "load_dates", track)
        run = develop(config)
    assert max(observed) == 39
    before = file_hash(run / "models.joblib")
    evaluate(run)
    assert file_hash(run / "models.joblib") == before
    return run


def test_complete_study_and_shared_denominators(study):
    summary = read_json(study / "evaluation/summary.json")
    assert summary["status"] == "complete"
    assert summary["source_kind"] == "synthetic"
    assert len(summary["metrics"]) == 6
    assert len({row["count"] for row in summary["metrics"]}) == 1
    assert len(summary["comparisons"]) == 2
    intervals = pl.read_parquet(study / "evaluation/interval_metrics.parquet")
    assert intervals["coverage"].is_between(0, 1).all()
    assert intervals["mean_width"].is_finite().all()
    assert len(set(intervals["count"])) == 1
    # Audit against original imported rows, not the transformed frame used by the scorer.
    from pathlib import Path

    dataset = Path(read_json(study / "config.json")["dataset"])
    forecasts = pl.concat([pl.read_parquet(p) for p in sorted(
        (study / "evaluation/predictions").glob("*.parquet"))])
    labels = load_dates(dataset, summary["test_dates"])
    joined = forecasts.join(labels.select("date_id", "stock_id", "seconds_in_bucket", "target"),
                            on=["date_id", "stock_id", "seconds_in_bucket"], validate="1:1")
    assert joined.height == labels.height
    valid = joined.filter(pl.col("target").is_not_null())
    expected = np.abs(valid["target"].to_numpy() - valid["regime"].to_numpy()).mean()
    actual = next(r["mae"] for r in summary["metrics"] if r["model"] == "regime")
    assert expected == pytest.approx(actual, abs=1e-12)


def test_fit_dates_and_unknown_stocks(study):
    models = joblib.load(study / "models.joblib")
    assert models["experts"].fit_dates == list(range(30))
    assert models["regimes"].fit_dates == list(range(30))
    config = Config.from_dict(read_json(study / "config.json"))
    from pathlib import Path
    frame = build_features(load_dates(Path(config.dataset), [45])).with_columns(pl.lit(999).alias("stock_id"))
    assert np.isfinite(models["experts"].predict(frame)).all()


def test_first_test_day_reveals_last_calibration_day(study):
    events = [json.loads(line) for line in (study / "evaluation/events.jsonl").read_text().splitlines()]
    assert events[0]["date_id"] == 40 and events[0]["revealed_date"] == 39
    assert events[0]["fixed_radii"] is not None
    assert events[0]["rolling_radii"]
    for previous, current in zip(events, events[1:]):
        assert current["revealed_date"] == previous["date_id"]
        assert current["revealed_regime"] == previous["regime"]


def test_evaluation_refuses_reuse_and_tampering(study):
    with pytest.raises(FileExistsError, match="already exists"):
        evaluate(study)
    path = study / "selection.json"
    original = path.read_bytes()
    try:
        path.write_text("{}")
        with pytest.raises(ValueError, match="artifact changed"):
            verify_run(study)
    finally:
        path.write_bytes(original)


def test_changed_scientific_code_is_rejected(study, monkeypatch):
    import regimeshift.pipeline as pipeline
    actual = pipeline.provenance()
    actual["source_hashes"]["adaptation.py"] = "modified"
    monkeypatch.setattr(pipeline, "provenance", lambda: actual)
    with pytest.raises(ValueError, match="Scientific code changed"):
        verify_execution_environment(study)


def test_causal_regime_assignments_and_expert_forecasts(study):
    from pathlib import Path

    from regimeshift.features import daily_summaries

    models = joblib.load(study / "models.joblib")
    dataset = Path(read_json(study / "config.json")["dataset"])
    previous = daily_summaries(build_features(load_dates(dataset, [39]))).row(0, named=True)
    raw = load_dates(dataset, [40, 41, 42, 43])
    changed = raw.with_columns(
        pl.when((pl.col("date_id") > 42) | ((pl.col("date_id") == 42)
                & (pl.col("seconds_in_bucket") > 30)))
        .then(pl.col("wap") * 2).otherwise(pl.col("wap")).alias("wap")
    )
    first, _ = cache_predictions(build_features(raw), models["experts"], models["regimes"], previous)
    second, _ = cache_predictions(build_features(changed), models["experts"], models["regimes"], previous)
    prior = (pl.col("date_id") < 42) | ((pl.col("date_id") == 42)
                                        & (pl.col("seconds_in_bucket") <= 30))
    assert first.filter(prior).equals(second.filter(prior))
    # Later observations on date 42 cannot change the regime assigned for ANY of date 42.
    assert first.filter(pl.col("date_id") == 42)["regime_id"].equals(
        second.filter(pl.col("date_id") == 42)["regime_id"])


def test_report_and_notebook_structure(study):
    import nbformat
    output = report(study, execute_notebook=True)
    assert "SYNTHETIC" in (output / "REPORT.md").read_text(encoding="utf-8")
    assert len(list(output.glob("*.png"))) == 5
    nb = nbformat.read(output / "research.ipynb", as_version=4)
    nbformat.validate(nb)
    assert any("## Takeaways" in c.source for c in nb.cells)
    images = [output for cell in nb.cells for output in cell.get("outputs", [])
              if "image/png" in output.get("data", {})]
    assert len(images) == 5
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == "code")
