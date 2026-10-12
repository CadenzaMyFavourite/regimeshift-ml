"""Explicit defaults and a deliberately small configuration surface."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    dataset: str = "data/processed/optiver"
    run_dir: str = "runs/optiver-v1"
    seed: int = 42
    threads: int = 4
    ridge_alphas: tuple = (1.0, 10.0, 100.0)
    lgbm_leaves: tuple = (15, 31)
    lgbm_estimators: int = 300
    lgbm_learning_rate: float = 0.05
    lgbm_min_child_samples: int = 100
    half_lives: tuple = (5, 10, 20)
    temperatures: tuple = (0.1, 0.5, 1.0)
    shrinkage_dates: float = 5.0
    interval_coverage: float = 0.9
    interval_window: int = 20
    bootstrap_block: int = 5
    bootstrap_reps: int = 2000

    def validate(self):
        for name in ("threads", "lgbm_estimators", "lgbm_min_child_samples", "interval_window",
                     "bootstrap_block", "bootstrap_reps"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        for name in ("ridge_alphas", "lgbm_leaves", "half_lives", "temperatures"):
            values = getattr(self, name)
            if not values or any(x <= 0 for x in values):
                raise ValueError(f"{name} must contain positive values")
        if any(x < 2 for x in self.lgbm_leaves):
            raise ValueError("lgbm_leaves must be >= 2")
        if not 0 < self.interval_coverage < 1 or self.shrinkage_dates <= 0:
            raise ValueError("Invalid coverage or shrinkage")
        if not 0 < self.lgbm_learning_rate <= 1:
            raise ValueError("Invalid learning rate")
        return self

    def to_dict(self):
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path: Path):
        values = yaml.safe_load(path.read_text()) or {}
        config = cls.from_dict(values)
        # Paths are relative to the invocation directory, explicitly documented.
        config.dataset = str(Path(config.dataset).resolve())
        config.run_dir = str(Path(config.run_dir).resolve())
        return config
