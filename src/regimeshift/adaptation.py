"""Discounted daily losses with shrinkage toward global expert skill."""

from collections import deque

import numpy as np


class AdaptiveWeights:
    def __init__(self, half_life=10, temperature=.5, shrinkage=5., regime_aware=True):
        if half_life <= 0 or temperature <= 0 or shrinkage <= 0:
            raise ValueError("Adaptation parameters must be positive")
        self.decay = 2 ** (-1 / half_life)
        self.temperature, self.shrinkage = temperature, shrinkage
        self.regime_aware = regime_aware
        self.global_sum = np.zeros(2)
        self.global_count = 0.
        self.regime_sum = np.zeros((3, 2))
        self.regime_count = np.zeros(3)

    def reveal(self, regime: int | None, normalized_losses: np.ndarray | None):
        # Called once per new observed trading date, including dates without labels.
        self.global_sum *= self.decay
        self.global_count *= self.decay
        self.regime_sum *= self.decay
        self.regime_count *= self.decay
        if normalized_losses is None:
            return
        losses = np.asarray(normalized_losses)
        if losses.shape != (2,) or not np.isfinite(losses).all() or (losses < 0).any():
            raise ValueError("Expected two finite nonnegative expert losses")
        self.global_sum += losses
        self.global_count += 1
        self.regime_sum[regime] += losses
        self.regime_count[regime] += 1

    def weights(self, regime: int) -> np.ndarray:
        if self.global_count <= 1e-12:
            return np.array([.5, .5])
        loss = self.global_sum / self.global_count
        count = self.regime_count[regime]
        if self.regime_aware and count > 1e-12:
            alpha = count / (count + self.shrinkage)
            loss = alpha * self.regime_sum[regime] / count + (1 - alpha) * loss
        logits = -loss / self.temperature
        unnormalized = np.exp(logits - logits.max())
        return unnormalized / unnormalized.sum()

    def snapshot(self):
        return {"global_loss_sum": self.global_sum.tolist(),
                "global_effective_dates": self.global_count,
                "regime_loss_sum": self.regime_sum.tolist(),
                "regime_effective_dates": self.regime_count.tolist()}


class ResidualIntervals:
    def __init__(self, coverage=.9, window=20):
        self.coverage = coverage
        self.days = deque(maxlen=window)
        self.fixed = None

    def reveal(self, date: int, residuals: dict[str, np.ndarray]):
        self.days.append((date, residuals))

    def radii(self):
        if not self.days:
            return {}
        result = {}
        for model in self.days[0][1]:
            values = np.concatenate([res[model] for _, res in self.days])
            if len(values):
                result[model] = float(np.quantile(values, self.coverage, method="higher"))
        return result

    def freeze(self, calibration_residuals: list[dict]):
        self.fixed = {}
        for model in calibration_residuals[0]:
            values = np.concatenate([res[model] for res in calibration_residuals])
            if not len(values):
                raise ValueError("Calibration block has no valid targets")
            self.fixed[model] = float(np.quantile(values, self.coverage, method="higher"))
