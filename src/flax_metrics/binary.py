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
        self.total.value += (logits > self.threshold) @ labels


class Precision(nnx.metrics.Average):
    """Precision metric, the fraction of identified positives that are true positives.

    Args:
        threshold: Threshold for identifying items as positives.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(self, *, logits: jnp.ndarray, labels: jnp.ndarray, **_) -> None:
        # The denominator is the number of identified positives.
        self.count.value += (logits > self.threshold).sum()
        # The numerator is the number of those that are actually positives.
        self.total.value += (logits > self.threshold) @ labels
