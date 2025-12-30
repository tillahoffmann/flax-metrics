from flax import nnx
from jax import numpy as jnp


class Recall(nnx.metrics.Average):
    """Recall metric, the fraction of actual positives that were correctly identified.

    Args:
        threshold: Threshold for identifying items as positives.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        # The denominator is the number of positives.
        self.count.value += labels.sum()
        # The numerator is the number of true positives.
        self.total.value += ((logits > self.threshold) * labels).sum()


class Precision(nnx.metrics.Average):
    """Precision metric, the fraction of identified positives that are true positives.

    Args:
        threshold: Threshold for identifying items as positives.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        predictions = logits > self.threshold
        # The denominator is the number of identified positives.
        self.count.value += predictions.sum()
        # The numerator is the number of those that are actually positives.
        self.total.value += (predictions * labels).sum()


class F1Score(nnx.Metric):
    """F1 score, the harmonic mean of precision and recall.

    Args:
        threshold: Threshold for identifying items as positives.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def reset(self) -> None:
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        predictions = logits > self.threshold
        self.true_positives.value += (predictions * labels).sum()
        self.actual_positives.value += labels.sum()
        self.predicted_positives.value += predictions.sum()

    def compute(self) -> jnp.ndarray:
        # F1 = 2 * TP / (2 * TP + FP + FN) = 2 * TP / (predicted + actual)
        return (
            2
            * self.true_positives.value
            / (self.predicted_positives.value + self.actual_positives.value)
        )
