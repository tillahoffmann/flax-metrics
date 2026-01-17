"""Metrics for evaluating classifiers, including recall, precision, and F1-score. These
metrics operate on logits and binary or multinomial labels, applying a threshold to
convert logits to point estimates where required.
"""

from typing import Self

from flax import nnx
from jax import numpy as jnp
from jax.nn import logsumexp, softplus
from jax.scipy.special import gammaln

from .base import Average, BaseMetric


class Recall(Average):
    """Recall metric, the fraction of actual positives that were correctly identified.

    .. seealso::
        This metric is implemented in scikit-learn as :func:`sklearn.metrics.recall_score`.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Recall
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = Recall()
        >>> metric.update(labels=labels, logits=logits)
        Recall(...)
        >>> metric.compute()
        Array(0.5, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(
        self,
        labels: jnp.ndarray,
        logits: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            labels: Ground truth binary labels.
            logits: Predicted logits.
            mask: Binary mask indicating which elements to include.
        """
        if mask is None:
            mask = jnp.ones_like(labels)
        # The denominator is the number of positives.
        self.count[...] += (labels * mask).sum()
        # The numerator is the number of true positives.
        self.total[...] += ((logits > self.threshold) * labels * mask).sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the recall."""
        return super().compute()


class Precision(Average):
    """Precision metric, the fraction of identified positives that are true positives.

    .. seealso::
        This metric is implemented in scikit-learn as :func:`sklearn.metrics.precision_score`.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import Precision
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = Precision()
        >>> metric.update(labels=labels, logits=logits)
        Precision(...)
        >>> metric.compute()
        Array(0.6666667, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        super().__init__()
        self.threshold = threshold

    def update(
        self,
        labels: jnp.ndarray,
        logits: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            labels: Ground truth binary labels.
            logits: Predicted logits.
            mask: Binary mask indicating which elements to include.
        """
        if mask is None:
            mask = jnp.ones_like(labels)
        predictions = logits > self.threshold
        # The denominator is the number of identified positives.
        self.count[...] += (predictions * mask).sum()
        # The numerator is the number of those that are actually positives.
        self.total[...] += (predictions * labels * mask).sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the precision."""
        return super().compute()


class F1Score(BaseMetric):
    """F1 score, the harmonic mean of precision and recall.

    .. seealso::
        This metric is implemented in scikit-learn as :func:`sklearn.metrics.f1_score`.

    Args:
        threshold: Threshold for identifying items as positives.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import F1Score
        >>>
        >>> labels = jnp.array([ 0,  0,  0,  1,  1,  1,  1])
        >>> logits = jnp.array([-1, -1,  1,  1,  1, -1, -1])
        >>> metric = F1Score()
        >>> metric.update(labels=labels, logits=logits)
        F1Score(...)
        >>> metric.compute()
        Array(0.5714286, dtype=float32)
    """

    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = threshold
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.true_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.actual_positives = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        self.predicted_positives = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )
        return self

    def update(
        self,
        labels: jnp.ndarray,
        logits: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            labels: Ground truth binary labels.
            logits: Predicted logits.
            mask: Binary mask indicating which elements to include.
        """
        if mask is None:
            mask = jnp.ones_like(labels)
        predictions = logits > self.threshold
        self.true_positives[...] += (predictions * labels * mask).sum()
        self.actual_positives[...] += (labels * mask).sum()
        self.predicted_positives[...] += (predictions * mask).sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the F1 score."""
        # F1 = 2 * TP / (2 * TP + FP + FN) = 2 * TP / (predicted + actual)
        return (
            2
            * self.true_positives[...]
            / (self.predicted_positives[...] + self.actual_positives[...])
        )


class Accuracy(Average):
    """Accuracy metric, the fraction of correct predictions.

    For multi-class classification, the logits are argmax-ed before comparing to labels.
    For binary classification, pass a ``threshold`` to determine positive predictions.

    .. seealso::
        This metric is implemented in scikit-learn as :func:`sklearn.metrics.accuracy_score`.

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


class LogProb(Average):
    """Log probability score, the mean likelihood of an outcome.

    The metric supports three modes:

    1. Binary classification if the :code:`logits` and :code:`labels` have shape
       :code:`(..., 1)`.
    2. Categorical classification if the inputs have shape :code:`(..., num_classes)`
       and the :code:`labels` are one-hot encoded, i.e.,
       :code:`labels.sum(axis=-1) == 1`.
    3. Multinomial outcomes if the inputs have shape :code:`(..., num_classes)` and the
       :code:`labels` are many-hot encoded, i.e., :code:`labels.sum(axis=-1) > 1`.

    Categorical and multinomial outcomes may be mixed within the same batch because
    multinomial outcomes with one sample are equivalent to categorical outcomes.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import LogProb
        >>>
        >>> labels = jnp.array([[ 0,  0,  0,  1,  1,  1,  1]])
        >>> logits = jnp.array([[-1, -1,  1,  1,  1, -1, -1]])
        >>> metric = LogProb()
        >>> metric.update(labels=labels, logits=logits)
        LogProb(...)
        >>> metric.compute()
        Array(-5.879968, dtype=float32)
    """

    def __init__(self) -> None:
        super().__init__()

    def update(
        self,
        labels: jnp.ndarray,
        logits: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the metric with a batch of predictions.

        Args:
            labels: Ground truth binary labels or multinomial counts with shape
                :code:`(..., num_classes)`, where :code:`...` denotes the batch shape.
                For binary classification, use labels with shape :code:`(..., 1)`.
            logits: Predicted logits with shape :code:`(..., num_classes)`, where
                :code:`...` denotes the batch shape. For binary classification, use
                logits with shape :code:`(..., 1)`.
            mask: Binary mask indicating which elements to include.
        """
        if logits.shape[-1] == 1:
            # Binary classification with likelihood based on (although using softplus)
            # https://github.com/pyro-ppl/numpyro/blob/6a1af1f4795d9b0b179e76ab05a13cc561dcecca/numpyro/distributions/util.py#L297-L300.
            log_prob = (logits * labels - softplus(logits)).squeeze(axis=-1)
        else:
            # Multinomial classification with likelihood based on
            # https://github.com/pyro-ppl/numpyro/blob/6a1af1f4795d9b0b179e76ab05a13cc561dcecca/numpyro/distributions/discrete.py#L699-L708.
            total = labels.sum(axis=-1)
            norm = total * logsumexp(logits, axis=-1) - gammaln(total + 1)
            log_prob = (
                jnp.sum(
                    labels * jnp.where(labels == 0, 0, logits) - gammaln(labels + 1),
                    axis=-1,
                )
                - norm
            )
        if mask is None:
            mask = jnp.ones_like(log_prob)
        self.total[...] += (log_prob * mask).sum()
        self.count[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the mean log probability score."""
        return super().compute()
