from typing import Callable, Self

from flax import nnx
from jax import numpy as jnp

from .base import BaseMetric


class LpError(BaseMetric):
    """L-p error metric with optional transform.

    Computes ``mean(|transform(target) - transform(prediction)|^p)^(1/p)`` if ``norm=True``,
    or ``mean(|transform(target) - transform(prediction)|^p)`` if ``norm=False``.

    Args:
        p: Exponent of the L-p metric.
        norm: Normalize the metric by raising to ``1 / p`` power.
        transform: Apply a transformation to the inputs element-wise before evaluating
            the metric. For example, use ``jnp.log1p`` for log-space errors.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import LpError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.0, 2.5])
        >>> metric = LpError(p=2, norm=False)
        >>> metric.update(target=target, prediction=prediction)
        LpError(...)
        >>> metric.compute()
        Array(0.166..., dtype=float32)
    """

    def __init__(
        self, p: float, *, norm: bool = True, transform: Callable | None = None
    ) -> None:
        self.p = p
        self.norm = norm
        self.transform = transform
        self.total = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state."""
        self.total = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        return self

    def update(
        self,
        target: jnp.ndarray,
        prediction: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            target: Ground truth values.
            prediction: Predicted values.
            mask: Binary mask indicating which elements to include.
        """
        if self.transform is not None:
            target = self.transform(target)
            prediction = self.transform(prediction)

        error = jnp.abs(target - prediction) ** self.p

        if mask is None:
            mask = jnp.ones_like(error)
        self.total[...] += (error * mask).sum()
        self.count[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the L-p error."""
        mean_error = self.total[...] / self.count[...]
        if self.norm:
            return mean_error ** (1 / self.p)
        return mean_error


class MeanAbsoluteError(LpError):
    """Mean absolute error (MAE).

    .. seealso::
        This metric is implemented in scikit-learn as
        :func:`sklearn.metrics.mean_absolute_error`.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import MeanAbsoluteError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.0, 2.5])
        >>> metric = MeanAbsoluteError()
        >>> metric.update(target=target, prediction=prediction)
        MeanAbsoluteError(...)
        >>> metric.compute()
        Array(0.333..., dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__(p=1, norm=True)


class MeanSquaredError(LpError):
    """Mean squared error (MSE).

    .. seealso::
        This metric is implemented in scikit-learn as
        :func:`sklearn.metrics.mean_squared_error`.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import MeanSquaredError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.0, 2.5])
        >>> metric = MeanSquaredError()
        >>> metric.update(target=target, prediction=prediction)
        MeanSquaredError(...)
        >>> metric.compute()
        Array(0.166..., dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__(p=2, norm=False)


class RootMeanSquaredError(LpError):
    """Root mean squared error (RMSE).

    .. seealso::
        This metric is implemented in scikit-learn as
        :func:`sklearn.metrics.root_mean_squared_error`.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import RootMeanSquaredError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.0, 2.5])
        >>> metric = RootMeanSquaredError()
        >>> metric.update(target=target, prediction=prediction)
        RootMeanSquaredError(...)
        >>> metric.compute()
        Array(0.408..., dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__(p=2, norm=True)


class MeanSquaredLogError(LpError):
    """Mean squared logarithmic error (MSLE).

    .. seealso::
        This metric is implemented in scikit-learn as
        :func:`sklearn.metrics.mean_squared_log_error`.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import MeanSquaredLogError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.5, 3.5])
        >>> metric = MeanSquaredLogError()
        >>> metric.update(target=target, prediction=prediction)
        MeanSquaredLogError(...)
        >>> metric.compute()
        Array(0.0291..., dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__(p=2, norm=False, transform=jnp.log1p)


class RootMeanSquaredLogError(LpError):
    """Root mean squared logarithmic error (RMSLE).

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import RootMeanSquaredLogError
        >>>
        >>> target = jnp.array([1.0, 2.0, 3.0])
        >>> prediction = jnp.array([1.5, 2.5, 3.5])
        >>> metric = RootMeanSquaredLogError()
        >>> metric.update(target=target, prediction=prediction)
        RootMeanSquaredLogError(...)
        >>> metric.compute()
        Array(0.170..., dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__(p=2, norm=True, transform=jnp.log1p)
