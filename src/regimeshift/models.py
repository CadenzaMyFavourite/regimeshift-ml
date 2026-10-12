"""Frozen experts: train-only preprocessing, tuning-only model selection."""

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy import sparse
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import Config
from .features import NUMERIC_FEATURES


@dataclass
class ExpertBundle:
    imputer: SimpleImputer
    scaler: StandardScaler
    encoder: OneHotEncoder
    stock_map: dict
    ridge: Ridge | None = None
    lightgbm: lgb.LGBMRegressor | None = None
    fit_dates: list | None = None
    threads: int = 4

    def matrices(self, frame):
        raw = frame.select(NUMERIC_FEATURES).to_numpy().astype(np.float64)
        numeric = self.scaler.transform(self.imputer.transform(raw))
        categorical = self.encoder.transform(frame.select("stock_id").to_numpy())
        ridge_x = sparse.hstack([sparse.csr_matrix(numeric), categorical], format="csr")
        stock_codes = np.array([self.stock_map.get(s, -1) for s in frame["stock_id"]])
        tree_x = np.column_stack([raw, stock_codes]).astype(np.float32)
        return ridge_x, tree_x

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        ridge_x, tree_x = self.matrices(frame)
        result = np.column_stack([
            self.ridge.predict(ridge_x),
            self.lightgbm.booster_.predict(tree_x, num_threads=self.threads),
        ])
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite expert predictions")
        return result


def fit_experts(train: pl.DataFrame, tune: pl.DataFrame, config: Config):
    train_y, tune_y = train["target"].to_numpy(), tune["target"].to_numpy()
    train_valid, tune_valid = np.isfinite(train_y), np.isfinite(tune_y)
    if not train_valid.any() or not tune_valid.any():
        raise ValueError("Training and tuning blocks each need finite targets")
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    numeric = imputer.fit_transform(train.select(NUMERIC_FEATURES).to_numpy())
    scaler = StandardScaler().fit(numeric)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True, dtype=np.float64)
    encoder.fit(train.select("stock_id").to_numpy())
    bundle = ExpertBundle(imputer, scaler, encoder,
                          {s: i for i, s in enumerate(sorted(train["stock_id"].unique()))},
                          fit_dates=sorted(train["date_id"].unique()), threads=config.threads)
    train_ridge, train_tree = bundle.matrices(train)
    tune_ridge, tune_tree = bundle.matrices(tune)
    scores, best_ridge, best_tree = [], np.inf, np.inf
    for alpha in config.ridge_alphas:
        model = Ridge(alpha=alpha, solver="lsqr", tol=1e-6)
        model.fit(train_ridge[train_valid], train_y[train_valid])
        mae = float(np.mean(np.abs(tune_y[tune_valid] - model.predict(tune_ridge[tune_valid]))))
        scores.append({"model": "ridge", "alpha": alpha, "tune_mae": mae})
        if mae < best_ridge:
            bundle.ridge, best_ridge = model, mae
    for leaves in config.lgbm_leaves:
        model = lgb.LGBMRegressor(
            objective="regression_l1", num_leaves=leaves,
            n_estimators=config.lgbm_estimators, learning_rate=config.lgbm_learning_rate,
            min_child_samples=config.lgbm_min_child_samples,
            random_state=config.seed, n_jobs=config.threads, verbosity=-1,
            deterministic=True, force_col_wise=True,
        )
        model.fit(train_tree[train_valid], train_y[train_valid],
                  categorical_feature=[len(NUMERIC_FEATURES)])
        predictions = model.booster_.predict(tune_tree[tune_valid], num_threads=config.threads)
        mae = float(np.mean(np.abs(tune_y[tune_valid] - predictions)))
        scores.append({"model": "lightgbm", "num_leaves": leaves, "tune_mae": mae})
        if mae < best_tree:
            bundle.lightgbm, best_tree = model, mae
    return bundle, scores, max(float(np.mean(np.abs(train_y[train_valid]))), 1e-8)
