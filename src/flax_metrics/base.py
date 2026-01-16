from typing import NamedTuple, Self

from flax import nnx
from jax import numpy as jnp


class Statistics(NamedTuple):
    """Statistics computed by the Welford metric."""

    mean: jnp.ndarray
    standard_error_of_mean: jnp.ndarray
    standard_deviation: jnp.ndarray


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

    def reset(self) -> Self:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Reset the state of the metric in-place.

        Returns:
            The metric instance.
        """
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

    def reset(self) -> Self:
        """Reset the metric state."""
        self.total = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        return self

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


class Accuracy(Average):
    """Accuracy metric, the fraction of correct predictions.

    For multi-class classification, the logits are argmax-ed before comparing to labels.
    For binary classification, pass a ``threshold`` to determine positive predictions.

    Args:
        threshold: For binary classification, logits >= threshold are considered
            positive. If None (default), multi-class classification is assumed.

    Example:

        Multi-class classification:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Accuracy
        >>>
        >>> logits = jnp.array([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        >>> labels = jnp.array([1, 1, 1])
        >>> metric = Accuracy()
        >>> metric.update(logits=logits, labels=labels)
        Accuracy(...)
        >>> metric.compute()
        Array(0.666..., dtype=float32)

        Binary classification:

        >>> logits = jnp.array([0.6, 0.4, 0.8, 0.3])
        >>> labels = jnp.array([1, 1, 1, 0])
        >>> metric = Accuracy(threshold=0.5)
        >>> metric.update(logits=logits, labels=labels)
        Accuracy(...)
        >>> metric.compute()
        Array(0.75, dtype=float32)
    """

    def __init__(self, threshold: float | None = None) -> None:
        super().__init__()
        self.threshold = threshold

    def update(
        self,
        logits: jnp.ndarray,
        labels: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            logits: Predicted logits. For multi-class, shape ``(..., num_classes)``.
                For binary, shape ``(...,)``.
            labels: Ground truth integer labels with shape ``(...,)``.
            mask: Binary mask indicating which elements to include.
        """
        if self.threshold is not None:
            # Binary classification
            predictions = logits >= self.threshold
            correct = predictions == (labels > 0)
        else:
            # Multi-class classification
            predictions = logits.argmax(axis=-1)
            correct = predictions == labels

        if mask is None:
            mask = jnp.ones_like(correct)
        self.total[...] += (correct * mask).sum()
        self.count[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the accuracy."""
        return super().compute()


class Welford(BaseMetric):
    """Welford metric, computing running mean and variance using Welford's algorithm.

    This is useful for computing statistics over a stream of data without storing all
    values in memory.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Welford
        >>>
        >>> values = jnp.array([1.0, 2.0, 3.0, 4.0])
        >>> metric = Welford()
        >>> metric.update(values=values)
        Welford(...)
        >>> metric.compute()
        Statistics(mean=Array(2.5, dtype=float32),
                   standard_error_of_mean=Array(0.559..., dtype=float32),
                   standard_deviation=Array(1.118..., dtype=float32))
    """

    def __init__(self) -> None:
        self.count = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.mean = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.m2 = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state."""
        self.count = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.mean = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.m2 = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        return self

    def update(
        self, values: jnp.ndarray, *_args, mask: jnp.ndarray | None = None, **_kwargs
    ) -> Self:
        """Update the metric with new values.

        Args:
            values: Array of values to include in the statistics.
            mask: Binary mask indicating which elements to include.
        """
        if mask is None:
            mask = jnp.ones_like(values)

        count = mask.sum()
        # Compute masked mean: sum(values * mask) / sum(mask)
        batch_mean = (values * mask).sum() / count
        # Compute masked variance: sum((values - mean)^2 * mask) / sum(mask)
        batch_var = ((values - batch_mean) ** 2 * mask).sum() / count

        original_count = self.count[...]
        self.count[...] += count
        delta = batch_mean - self.mean[...]
        self.mean[...] += delta * count / self.count[...]
        m2 = batch_var * count
        self.m2[...] += m2 + delta * delta * count * original_count / self.count[...]
        return self

    def compute(self) -> Statistics:
        """Compute and return the mean and variance statistics."""
        variance = self.m2[...] / self.count[...]
        standard_deviation = variance**0.5
        sem = standard_deviation / (self.count[...] ** 0.5)
        return Statistics(
            mean=self.mean[...],
            standard_error_of_mean=sem,
            standard_deviation=standard_deviation,
        )
