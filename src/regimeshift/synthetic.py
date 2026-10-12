"""Synthetic microstructure-shaped data; never evidence about real markets."""

from pathlib import Path

import numpy as np
import polars as pl

from .utils import write_json


def generate(path: Path, dates: int = 160, stocks: int = 12, snapshots: int = 24,
             seed: int = 42) -> Path:
    if path.exists():
        raise FileExistsError(f"Synthetic file already exists: {path}")
    if dates < 50 or stocks < 2 or not 2 <= snapshots <= 55:
        raise ValueError("Use >=50 dates, >=2 stocks and 2..55 snapshots")
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(dates):
        state = (d // 12) % 3
        for stock in range(stocks):
            wap = 1.0
            for step in range(snapshots):
                signal = rng.uniform(-.9, .9)
                liquidity = [1200., 3000., 700.][state] * rng.lognormal(0, .15)
                spread = [2., 5., 10.][state] * rng.lognormal(0, .08) / 1e4
                wap *= 1 + rng.normal(0, [0.5, 1., 2.][state]) / 1e4
                bid, ask = wap - spread / 2, wap + spread / 2
                # Changing functional relationships plus a modest late scale shift.
                # These labels are simulated, not returns computed from future WAP.
                mean = [3 * signal, 8 * signal ** 3, 4 * np.sin(4 * signal)][state]
                target = mean * (1.0 if d < dates * .8 else 1.15) + rng.normal(0, .8)
                rows.append({
                    "date_id": d, "stock_id": stock, "seconds_in_bucket": step * 10,
                    "imbalance_size": abs(signal) * liquidity,
                    "imbalance_buy_sell_flag": float(np.sign(signal)),
                    "reference_price": 1.0, "matched_size": liquidity * 3,
                    "far_price": None if step < 12 else wap + signal / 1e4,
                    "near_price": None if step < 8 else wap + signal / 2e4,
                    "bid_price": bid, "ask_price": ask,
                    "bid_size": liquidity * (1 + signal) / 2,
                    "ask_size": liquidity * (1 - signal) / 2, "wap": wap,
                    "target": None if rng.random() < .005 else target,
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).sort(["date_id", "seconds_in_bucket", "stock_id"]).write_csv(path)
    write_json(path.with_suffix(".synthetic.json"), {
        "source_kind": "synthetic", "seed": seed, "dates": dates, "stocks": stocks,
        "snapshots": snapshots, "regime_period_dates": 12,
        "warning": "Simulated labels do not reproduce the Optiver target construction.",
    })
    return path
