import copy

import joblib
import numpy as np
import pytest

from regimeshift.adaptation import AdaptiveWeights, ResidualIntervals
from regimeshift.replay import Outcome, ReplayEngine

PARAMS = {"half_life": 5, "temperature": .5}


def replay(targets, regimes=None):
    engine = ReplayEngine(PARAMS, PARAMS, .5, 1.)
    records = []
    regimes = regimes or [0] * len(targets)
    for date, (target, regime) in enumerate(zip(targets, regimes)):
        experts = np.array([[0., 10.]])
        predictions, event = engine.predict(date, regime, experts)
        records.append((copy.deepcopy(predictions), copy.deepcopy(event)))
        engine.queue_labels(Outcome(date, regime, experts, predictions,
                                    np.array([target]), "calibration"))
    return records, engine


def test_current_and_future_labels_do_not_affect_past():
    original, _ = replay([0., 10., 0., 10.])
    changed, _ = replay([0., -1000., 333., 777.])
    for i in [0, 1]:
        for model in original[i][0]:
            np.testing.assert_array_equal(original[i][0][model], changed[i][0][model])
        assert original[i][1] == changed[i][1]
    assert original[2][1]["global_ridge_weight"] != changed[2][1]["global_ridge_weight"]


def test_reveal_attribution_uses_prediction_regime():
    results, _ = replay([0., 10., 5.], [0, 1, 2])
    event = results[1][1]
    assert event["regime"] == 1 and event["revealed_regime"] == 0
    assert event["after_reveal"]["regime"]["regime_effective_dates"] == [1., 0., 0.]
    assert results[0][1]["regime_ridge_weight"] == .5


def test_missing_labels_decay_without_fake_zero_loss():
    weights = AdaptiveWeights(half_life=1)
    weights.reveal(0, np.array([1., 5.]))
    expected = weights.weights(0)
    weights.reveal(1, None)
    assert weights.global_count == .5
    np.testing.assert_allclose(weights.weights(1), weights.weights(2))
    assert weights.global_sum.tolist() == [.5, 2.5]
    assert expected[0] > .5


def test_shrinkage_and_staleness():
    weights = AdaptiveWeights(half_life=1, shrinkage=5)
    weights.reveal(0, np.array([0., 1.]))
    weights.reveal(1, np.array([1., 0.]))
    assert weights.regime_count.tolist() == [.5, 1., 0.]
    assert weights.weights(0)[0] > weights.weights(2)[0]
    assert weights.weights(1)[0] < weights.weights(2)[0]


def test_controlled_shift_response_is_delayed():
    results, _ = replay([0.] * 8 + [10.] * 8)
    assert results[7][1]["global_ridge_weight"] > .9
    assert results[8][1]["global_ridge_weight"] > .9  # new regime's first label is not revealed
    assert results[-1][1]["global_ridge_weight"] < .5


def test_interval_quantiles_and_empty_date_window():
    interval = ResidualIntervals(coverage=.9, window=2)
    interval.reveal(0, {"ridge": np.array([1., 2., 3.])})
    assert interval.radii() == {"ridge": 3.}
    interval.reveal(1, {"ridge": np.array([4.])})
    interval.reveal(2, {"ridge": np.array([])})
    assert interval.radii() == {"ridge": 4.}
    interval.reveal(3, {"ridge": np.array([])})
    assert interval.radii() == {}


def test_resume_matches_uninterrupted(tmp_path):
    _, engine = replay([0., 10., 0.])
    path = tmp_path / "state.joblib"
    joblib.dump(engine, path)
    restored = joblib.load(path)
    experts = np.array([[3., 4.]])
    one, event_one = engine.predict(3, 1, experts)
    two, event_two = restored.predict(3, 1, experts)
    assert event_one == event_two
    for name in one:
        np.testing.assert_array_equal(one[name], two[name])


def test_replay_rejects_out_of_order_or_unpaired_events():
    _, engine = replay([0.])
    with pytest.raises(ValueError, match="strictly increase"):
        engine.predict(0, 0, np.array([[0., 10.]]))
    engine.predict(1, 0, np.array([[0., 10.]]))
    with pytest.raises(RuntimeError, match="Queue outcomes"):
        engine.predict(2, 0, np.array([[0., 10.]]))
