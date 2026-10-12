"""Replay has an explicit predict-then-queue-labels boundary."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from .adaptation import AdaptiveWeights, ResidualIntervals
from .data import KEYS

MODELS = ["zero", "ridge", "lightgbm", "fixed", "global", "regime"]


@dataclass
class Outcome:
    date: int
    regime: int
    experts: np.ndarray
    predictions: dict
    targets: np.ndarray
    phase: str

    def residuals(self):
        valid = np.isfinite(self.targets)
        return {name: np.abs(self.targets[valid] - self.predictions[name][valid])
                for name in MODELS}


class ReplayEngine:
    def __init__(self, global_params, regime_params, fixed_weight, loss_scale,
                 shrinkage=5., coverage=.9, interval_window=20):
        self.global_weights = AdaptiveWeights(**global_params, shrinkage=shrinkage,
                                              regime_aware=False)
        self.regime_weights = AdaptiveWeights(**regime_params, shrinkage=shrinkage,
                                              regime_aware=True)
        self.fixed_weight = fixed_weight
        self.loss_scale = loss_scale
        self.intervals = ResidualIntervals(coverage, interval_window)
        self.pending: Outcome | None = None
        self.last_date = None
        self.awaiting_labels = False

    def predict(self, date: int, regime: int, experts: np.ndarray):
        """No current targets enter this interface. Reveal only the queued prior date."""
        if self.awaiting_labels:
            raise RuntimeError("Queue outcomes after saving the preceding predictions")
        if self.last_date is not None and date <= self.last_date:
            raise ValueError("Replay dates must strictly increase")
        experts = np.asarray(experts, dtype=float)
        if experts.ndim != 2 or experts.shape[1] != 2 or not np.isfinite(experts).all():
            raise ValueError("Expected finite Ridge/LightGBM predictions")
        if regime not in (0, 1, 2):
            raise ValueError("Regime must be 0, 1 or 2")
        before = {"global": self.global_weights.snapshot(),
                  "regime": self.regime_weights.snapshot()}
        losses, revealed_date, revealed_regime = None, None, None
        if self.pending is not None:
            old = self.pending
            revealed_date, revealed_regime = old.date, old.regime
            valid = np.isfinite(old.targets)
            if valid.any():
                losses = np.mean(np.abs(old.experts[valid] - old.targets[valid, None]), axis=0)
                losses /= self.loss_scale
            if old.phase in {"calibration", "test"}:
                self.intervals.reveal(old.date, old.residuals())
        self.global_weights.reveal(revealed_regime, losses)
        self.regime_weights.reveal(revealed_regime, losses)
        global_w = self.global_weights.weights(regime)
        regime_w = self.regime_weights.weights(regime)
        predictions = {
            "zero": np.zeros(len(experts)), "ridge": experts[:, 0], "lightgbm": experts[:, 1],
            "fixed": experts @ np.array([self.fixed_weight, 1 - self.fixed_weight]),
            "global": experts @ global_w, "regime": experts @ regime_w,
        }
        self.last_date, self.awaiting_labels = date, True
        event = {
            "date_id": date, "regime": regime, "revealed_date": revealed_date,
            "revealed_regime": revealed_regime,
            "normalized_revealed_losses": None if losses is None else losses.tolist(),
            "global_ridge_weight": float(global_w[0]), "regime_ridge_weight": float(regime_w[0]),
            "before_reveal": before,
            "after_reveal": {"global": self.global_weights.snapshot(),
                             "regime": self.regime_weights.snapshot()},
            "rolling_radii": self.intervals.radii(), "fixed_radii": self.intervals.fixed,
        }
        self.pending = None
        return predictions, event

    def queue_labels(self, outcome: Outcome):
        if not self.awaiting_labels or outcome.date != self.last_date:
            raise RuntimeError("An outcome must follow that date's prediction")
        if len(outcome.targets) != len(outcome.experts):
            raise ValueError("Outcome rows are not aligned")
        self.pending, self.awaiting_labels = outcome, False


def prediction_frame(keys, predictions, event, next_date, phase):
    result = keys.select(KEYS).with_columns(
        pl.lit(event["regime"]).cast(pl.Int32).alias("regime_id"),
        pl.lit(next_date, dtype=pl.Int32).alias("label_reveal_date"),
        pl.lit(phase).alias("phase"),
        pl.lit(event["global_ridge_weight"]).alias("global_ridge_weight"),
        pl.lit(event["regime_ridge_weight"]).alias("regime_ridge_weight"),
        *[pl.Series(name, predictions[name]) for name in MODELS],
    )
    for kind, radii in [("fixed", event["fixed_radii"]), ("rolling", event["rolling_radii"])]:
        for name in MODELS:
            radius = (radii or {}).get(name)
            result = result.with_columns(pl.lit(radius, dtype=pl.Float64)
                                         .alias(f"{name}_{kind}_radius"))
    return result


def simulate_cached(cache: pl.DataFrame, global_params, regime_params, fixed_weight,
                    loss_scale, shrinkage=5.):
    """Independent cold-start replay for one tuning candidate (never training residuals)."""
    engine = ReplayEngine(global_params, regime_params, fixed_weight, loss_scale, shrinkage)
    totals, count = {m: 0. for m in MODELS}, 0
    for (date,), day in cache.partition_by("date_id", as_dict=True, maintain_order=True).items():
        regime = int(day["regime_id"][0])
        experts = day.select(["ridge", "lightgbm"]).to_numpy()
        predictions, _ = engine.predict(date, regime, experts)
        targets = day["target"].to_numpy()
        valid = np.isfinite(targets)
        count += int(valid.sum())
        for name in MODELS:
            totals[name] += float(np.abs(targets[valid] - predictions[name][valid]).sum())
        engine.queue_labels(Outcome(date, regime, experts, predictions, targets, "tune"))
    if not count:
        raise ValueError("No finite tuning targets")
    return {m: value / count for m, value in totals.items()}
