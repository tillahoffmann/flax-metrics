import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from numpy.testing import assert_almost_equal

from flax_metrics import Average


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 2.5),
        ([0.0, 0.0, 0.0], 0.0),
        ([5.0], 5.0),
        ([-1.0, 1.0], 0.0),
        ([1.5, 2.5, 3.5], 2.5),
    ],
)
def test_average_computes_mean(values, expected, jit, masked):
    """Verify Average computes the arithmetic mean."""
    values = jnp.array(values)

    metric = Average()
    if masked:
        validate_masking(metric, (values,), {}, jit=jit, event_dim=0)
        return

    update, compute = update_and_compute(metric, jit)
    update(values)
    actual = float(compute())

    assert_almost_equal(actual, expected)


def test_average_accumulation(jit):
    """Accumulated average over batches matches numpy on combined data."""
    metric = Average()
    update, compute = update_and_compute(metric, jit)

    values1 = jnp.array([1.0, 2.0])
    update(values1)

    values2 = jnp.array([3.0, 4.0, 5.0])
    update(values2)

    actual = float(compute())

    all_values = np.concatenate([values1, values2])
    expected = np.mean(all_values)

    assert_almost_equal(actual, expected)


def test_average_reset(jit):
    """Reset clears accumulated state."""
    metric = Average()
    update, compute = update_and_compute(metric, jit)

    update(jnp.array([100.0, 200.0]))
    metric.reset()
    update(jnp.array([1.0, 2.0, 3.0]))

    actual = float(compute())
    expected = 2.0

    assert_almost_equal(actual, expected)
