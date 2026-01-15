import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from jax.scipy.special import expit, softmax
from numpy.testing import assert_almost_equal
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score

from flax_metrics import F1Score, LogProb, Precision, Recall


def _sklearn_negative_log_loss(labels, logits):
    """Compute negative log loss, handling binary vs multiclass shapes."""
    labels = np.asarray(labels)
    logits = np.asarray(logits)
    if logits.shape[-1] == 1:
        # Binary: use sigmoid for probabilities
        probs = expit(logits.squeeze(axis=-1))
        labels = labels.squeeze(axis=-1)
    else:
        # Multiclass: use softmax for probabilities
        probs = softmax(logits, axis=-1)
    return -log_loss(labels, probs)


METRICS = [
    (Recall, recall_score),
    (Precision, precision_score),
    (F1Score, f1_score),
]


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
@pytest.mark.parametrize(
    "logits, labels, threshold",
    [
        ([1.0, 1.0, -1.0, -1.0], [1, 1, 0, 0], 0.0),
        ([-1.0, -1.0, 1.0, 1.0], [1, 1, 0, 0], 0.0),
        ([1.0, -1.0, -1.0, 1.0], [1, 1, 0, 0], 0.0),
        ([0.6, 0.4, 0.3, 0.8], [1, 1, 0, 0], 0.5),
        ([0.5, 0.6], [1, 0], 0.5),
    ],
)
def test_binary_metric_matches_sklearn(
    metric_cls, sklearn_fn, logits, labels, threshold, jit, masked
):
    """Verify our metrics match sklearn."""
    logits = jnp.array(logits)
    labels = jnp.array(labels)

    metric = metric_cls(threshold=threshold)
    if masked:
        validate_masking(
            metric, (), {"logits": logits, "labels": labels}, jit=jit, event_dim=0
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(logits=logits, labels=labels)
    actual = float(compute())

    predictions = (logits > threshold).astype(int)
    expected = sklearn_fn(labels, predictions)

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "metric_cls, sklearn_fn",
    [
        (LogProb, _sklearn_negative_log_loss),
    ],
)
@pytest.mark.parametrize(
    "logits, labels",
    [
        # One-hot classification over four classes.
        ([[1.0, 1.0, -1.0, -1.0]], [[1, 0, 0, 0]]),
        # Batch of three binary classification.
        ([[-3], [1], [2.5]], [[0], [1], [1]]),
        # Extreme logits for numerical stability.
        ([[100.0], [-100.0]], [[1], [0]]),
        ([[100.0, -100.0, 0.0]], [[1, 0, 0]]),
        # Case with negative probabilities.
        ([[2.0, -jnp.inf, 1.0, -jnp.inf, -jnp.inf]], [[1, 0, 0, 0, 0]]),
    ],
)
def test_multinomial_metric_matches_sklearn(
    metric_cls, sklearn_fn, logits, labels, jit, masked
):
    logits = jnp.asarray(logits)
    labels = jnp.asarray(labels)

    metric = metric_cls()
    if masked:
        validate_masking(
            metric, (), {"logits": logits, "labels": labels}, jit=jit, event_dim=1
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(logits=logits, labels=labels)
    actual = float(compute())

    expected = sklearn_fn(labels, logits)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_accumulation_matches_sklearn(metric_cls, sklearn_fn, jit):
    """Accumulated metric over batches matches sklearn on combined data."""
    metric = metric_cls(threshold=0.0)
    update, compute = update_and_compute(metric, jit)

    logits1 = jnp.array([1.0, -1.0])
    labels1 = jnp.array([1, 0])
    update(logits=logits1, labels=labels1)

    logits2 = jnp.array([1.0, 1.0, -1.0])
    labels2 = jnp.array([1, 0, 0])
    update(logits=logits2, labels=labels2)

    actual = float(compute())

    all_logits = jnp.concatenate([logits1, logits2])
    all_labels = jnp.concatenate([labels1, labels2])
    predictions = (all_logits > 0.0).astype(int)
    expected = sklearn_fn(all_labels, predictions)

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_reset(metric_cls, sklearn_fn, jit):
    """Reset clears accumulated state."""
    metric = metric_cls(threshold=0.0)
    update, compute = update_and_compute(metric, jit)

    update(logits=jnp.array([1.0, 1.0]), labels=jnp.array([1, 1]))
    metric.reset()
    update(logits=jnp.array([-1.0, 1.0]), labels=jnp.array([1, 0]))

    actual = float(compute())

    predictions = jnp.array([0, 1])
    expected = sklearn_fn([1, 0], predictions)

    assert_almost_equal(actual, expected)
