from typing import Self

from flax import nnx
from jax import numpy as jnp


class BaseMetric(nnx.Metric):
    """Base class for Flax Metrics implementations.

    We inherit from :class:`flax.nnx.metrics.Metrics` to support :code:`isinstance` type
    checks. This class overrides :meth:`update` to accept positional and keyword
    arguments and a :code:`mask` parameter. :meth:`update` also returns :code:`Self` so
    :meth:`update`\\s and :meth:`compute` can be chained.
    """

    def update(self, *args, mask: jnp.ndarray | None = None, **kwargs) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the metric in-place.

        Args:
            *args: Positional arguments.
            mask: Binary mask indicating which elements to include.
            **kwargs: Keyword arguments.

        Returns:
            The metric instance.
        """
        # We override this method to:
        #   (a) allow arbitrary positional and keyword arguments.
        #   (b) return the metric, allowing for chaining.
        raise NotImplementedError


class Average(BaseMetric):
    """Average metric, the arithmetic mean of values.

    Args:
        argname: Name of the keyword argument to average.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Average
        >>>
        >>> values = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> metric = Average()
        >>> metric.update(values=values)
        Average(...)
        >>> metric.compute()
        Array(2.5, dtype=float32)
    """

    def __init__(self):
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> None:
        """Reset the metric state."""
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def update(
        self, values: jnp.ndarray, *_args, mask: jnp.ndarray | None = None, **_kwargs
    ) -> Self:
        if mask is None:
            mask = jnp.ones_like(values)
        self.total[...] += (values * mask).sum()
        self.count[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the average."""
        return self.total[...] / self.count[...]
