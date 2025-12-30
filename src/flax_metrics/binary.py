from flax import nnx
from jax import numpy as jnp


class Recall(nnx.metrics.Average):
    """Recall metric, the fraction of actual positives that were correctly identified.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Recall
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = Recall()
        >>> metric.update(logits=logits, labels=labels)
        >>> metric.compute()
        Array(0.5, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of predictions.

        Args:
            logits: Predicted logits.
            labels: Ground truth binary labels.
        """
        # The denominator is the number of positives.
        self.count.value += labels.sum()
        # The numerator is the number of true positives.
        self.total.value += ((logits > self.threshold) * labels).sum()

    def compute(self) -> jnp.ndarray:
        """Compute and return the recall."""
        return super().compute()


class Precision(nnx.metrics.Average):
    """Precision metric, the fraction of identified positives that are true positives.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Precision
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = Precision()
        >>> metric.update(logits=logits, labels=labels)
        >>> metric.compute()
        Array(0.6666667, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of predictions.

        Args:
            logits: Predicted logits.
            labels: Ground truth binary labels.
        """
        predictions = logits > self.threshold
        # The denominator is the number of identified positives.
        self.count.value += predictions.sum()
        # The numerator is the number of those that are actually positives.
        self.total.value += (predictions * labels).sum()

    def compute(self) -> jnp.ndarray:
        """Compute and return the precision."""
        return super().compute()


class F1Score(nnx.Metric):
    """F1 score, the harmonic mean of precision and recall.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import F1Score
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = F1Score()
        >>> metric.update(logits=logits, labels=labels)
        >>> metric.compute()
        Array(0.5714286, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of predictions.

        Args:
            logits: Predicted logits.
            labels: Ground truth binary labels.
        """
        predictions = logits > self.threshold
        self.true_positives.value += (predictions * labels).sum()
        self.actual_positives.value += labels.sum()
        self.predicted_positives.value += predictions.sum()

    def compute(self) -> jnp.ndarray:
        """Compute and return the F1 score."""
        # F1 = 2 * TP / (2 * TP + FP + FN) = 2 * TP / (predicted + actual)
        return (
            2
            * self.true_positives.value
            / (self.predicted_positives.value + self.actual_positives.value)
        )
