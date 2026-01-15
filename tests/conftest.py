"""Pytest configuration and fixtures for flax-metrics tests."""

import functools
import math
from typing import Sequence

import pytest
from flax import nnx
from jax import numpy as jnp
from jax import random
from numpy.testing import assert_almost_equal


@pytest.fixture(params=[False, True], ids=["eager", "jit"])
def jit(request):
    """Fixture that runs each test twice: once eager, once JIT-compiled."""
    return request.param


@pytest.fixture(params=[False, True], ids=["unmasked", "masked"])
def masked(request):
    """Fixture that runs each test twice: once with masked entries, once without."""
    return request.param


def update_and_compute(metric, jit):
    """Return update and compute functions, optionally JIT-compiled.

    For JIT compilation with NNX, the metric must be passed as an explicit
    argument to the jitted function (not captured via closure). We create
    wrapper functions that take kwargs and pass the metric explicitly.

    Args:
        metric: An nnx.Metric instance.
        jit: If True, return JIT-compiled versions of update and compute.

    Returns:
        Tuple of (update_fn, compute_fn) that can be called to update/compute.
    """
    if jit:

        @nnx.jit
        def jitted_update(m, *args, **kwargs):
            m.update(*args, **kwargs)

        @nnx.jit
        def jitted_compute(m):
            return m.compute()

        return functools.partial(jitted_update, metric), functools.partial(
            jitted_compute, metric
        )
    return metric.update, metric.compute


def validate_masking(
    metric, args: Sequence, kwargs: dict, *, jit: bool, event_dim: int
) -> None:
    # Construct a random mask with at least one positive element. The shape is inferred
    # based on the first positional argument to be passed to the metric. If there are no
    # positional argument, the first keyword argument is used.
    if args:
        mask_shape = args[0].shape
    elif kwargs:
        mask_shape = next(iter(kwargs.values())).shape
    else:
        raise ValueError("Cannot infer mask shape.")
    if event_dim:
        assert event_dim <= len(mask_shape)
        mask_shape = mask_shape[:-event_dim]

    # If there is no mask shape, exit early because we cannot mask scalars.
    if not mask_shape:
        return

    # Create a random mask with 50-50 split of masked entries.
    key = random.key(42)
    size = math.prod(mask_shape)
    mask = random.permutation(key, (jnp.arange(size) % 2 == 0)).reshape(mask_shape)
    assert mask.any(), "Invalid empty mask."

    update, compute = update_and_compute(metric, jit)

    metric.reset()
    update(
        *(arg[mask] for arg in args),
        **{key: value[mask] for key, value in kwargs.items()},
        mask=None,
    )
    expected = compute()

    metric.reset()
    update(*args, **kwargs, mask=mask)
    actual = compute()

    assert_almost_equal(actual, expected)
