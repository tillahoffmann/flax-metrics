"""Pytest configuration and fixtures for flax-metrics tests."""

import functools

import pytest
from flax import nnx


@pytest.fixture(params=[False, True], ids=["eager", "jit"])
def jit(request):
    """Fixture that runs each test twice: once eager, once JIT-compiled."""
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
        def jitted_update(m, **kwargs):
            m.update(**kwargs)

        @nnx.jit
        def jitted_compute(m):
            return m.compute()

        return functools.partial(jitted_update, metric), functools.partial(
            jitted_compute, metric
        )
    return metric.update, metric.compute
