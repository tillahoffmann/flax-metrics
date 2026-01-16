import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from numpy.testing import assert_almost_equal
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
    root_mean_squared_error,
    root_mean_squared_log_error,
)

from flax_metrics import (
    LpError,
    MeanAbsoluteError,
    MeanSquaredError,
    MeanSquaredLogError,
    RootMeanSquaredError,
    RootMeanSquaredLogError,
)

METRICS = [
    (MeanAbsoluteError, mean_absolute_error),
    (MeanSquaredError, mean_squared_error),
    (RootMeanSquaredError, root_mean_squared_error),
    (MeanSquaredLogError, mean_squared_log_error),
    (RootMeanSquaredLogError, root_mean_squared_log_error),
]


@pytest.mark.parametrize("metric_cls, sklearn_fn", METRICS)
@pytest.mark.parametrize(
    "target, prediction",
    [
        ([1.0, 2.0, 3.0], [1.5, 2.0, 2.5]),
        ([0.0, 1.0, 2.0, 3.0], [0.5, 1.5, 2.5, 3.5]),
        ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([10.0, 20.0], [12.0, 18.0]),
    ],
)
def test_metric_matches_sklearn(
    metric_cls, sklearn_fn, target, prediction, jit, masked
):
    """Verify our metrics match sklearn."""
    target = jnp.array(target)
    prediction = jnp.array(prediction)

    metric = metric_cls()
    if masked:
        validate_masking(
            metric,
            (),
            {"target": target, "prediction": prediction},
            jit=jit,
            event_dim=0,
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(target=target, prediction=prediction)
    actual = float(compute())

    expected = sklearn_fn(np.array(target), np.array(prediction))

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls, sklearn_fn", METRICS)
def test_metric_accumulation_matches_sklearn(metric_cls, sklearn_fn, jit):
    """Accumulated metric over batches matches sklearn on combined data."""
    metric = metric_cls()
    update, compute = update_and_compute(metric, jit)

    target1 = jnp.array([1.0, 2.0])
    prediction1 = jnp.array([1.5, 2.5])
    update(target=target1, prediction=prediction1)

    target2 = jnp.array([3.0, 4.0, 5.0])
    prediction2 = jnp.array([2.5, 4.0, 5.5])
    update(target=target2, prediction=prediction2)

    actual = float(compute())

    all_target = np.concatenate([target1, target2])
    all_prediction = np.concatenate([prediction1, prediction2])
    expected = sklearn_fn(all_target, all_prediction)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls, sklearn_fn", METRICS)
def test_metric_reset(metric_cls, sklearn_fn, jit):
    """Reset clears accumulated state."""
    metric = metric_cls()
    update, compute = update_and_compute(metric, jit)

    update(target=jnp.array([100.0, 200.0]), prediction=jnp.array([0.0, 0.0]))
    metric.reset()
    update(target=jnp.array([1.0, 2.0]), prediction=jnp.array([1.5, 2.5]))

    actual = float(compute())

    expected = sklearn_fn([1.0, 2.0], [1.5, 2.5])

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("p", [1, 2, 3])
def test_lp_error_matches_numpy(p, jit, masked):
    """Verify LpError matches numpy computation."""
    target = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    prediction = jnp.array([1.2, 1.8, 3.5, 3.9, 5.1])

    metric = LpError(p=p, norm=True)
    if masked:
        validate_masking(
            metric,
            (),
            {"target": target, "prediction": prediction},
            jit=jit,
            event_dim=0,
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(target=target, prediction=prediction)
    actual = float(compute())

    expected = np.mean(np.abs(target - prediction) ** p) ** (1 / p)
    assert_almost_equal(actual, expected, decimal=5)
