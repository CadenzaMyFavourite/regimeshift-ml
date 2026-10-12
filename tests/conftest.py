import os

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import polars as pl
import pytest

from regimeshift.data import validate_frame
from regimeshift.synthetic import generate


@pytest.fixture(scope="session")
def raw_frame(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    generate(path, dates=50, stocks=4, snapshots=8, seed=71)
    frame, _ = validate_frame(pl.read_csv(path))
    return frame
