import jax.numpy as jnp
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import f1_score, precision_score, recall_score

from flax_metrics import F1Score, Precision, Recall

METRICS = [
    (Recall, recall_score),
    (Precision, precision_score),
    (F1Score, f1_score),
]


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
@pytest.mark.parametrize(
    "logits,labels,threshold",
    [
        ([1.0, 1.0, -1.0, -1.0], [1, 1, 0, 0], 0.0),
        ([-1.0, -1.0, 1.0, 1.0], [1, 1, 0, 0], 0.0),
        ([1.0, -1.0, -1.0, 1.0], [1, 1, 0, 0], 0.0),
        ([0.6, 0.4, 0.3, 0.8], [1, 1, 0, 0], 0.5),
        ([0.5, 0.6], [1, 0], 0.5),
    ],
)
def test_metric_matches_sklearn(metric_cls, sklearn_fn, logits, labels, threshold):
    """Verify our metrics match sklearn."""
    logits = jnp.array(logits)
    labels = jnp.array(labels)

    metric = metric_cls(threshold=threshold)
    metric.update(logits=logits, labels=labels)
    actual = float(metric.compute())

    predictions = (logits > threshold).astype(int)
    expected = sklearn_fn(labels, predictions)

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_accumulation_matches_sklearn(metric_cls, sklearn_fn):
    """Accumulated metric over batches matches sklearn on combined data."""
    metric = metric_cls(threshold=0.0)

    logits1 = jnp.array([1.0, -1.0])
    labels1 = jnp.array([1, 0])
    metric.update(logits=logits1, labels=labels1)

    logits2 = jnp.array([1.0, 1.0, -1.0])
    labels2 = jnp.array([1, 0, 0])
    metric.update(logits=logits2, labels=labels2)

    actual = float(metric.compute())

    all_logits = jnp.concatenate([logits1, logits2])
    all_labels = jnp.concatenate([labels1, labels2])
    predictions = (all_logits > 0.0).astype(int)
    expected = sklearn_fn(all_labels, predictions)

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_reset(metric_cls, sklearn_fn):
    """Reset clears accumulated state."""
    metric = metric_cls(threshold=0.0)

    metric.update(logits=jnp.array([1.0, 1.0]), labels=jnp.array([1, 1]))
    metric.reset()
    metric.update(logits=jnp.array([-1.0, 1.0]), labels=jnp.array([1, 0]))

    actual = float(metric.compute())

    predictions = jnp.array([0, 1])
    expected = sklearn_fn([1, 0], predictions)

    assert_almost_equal(actual, expected)
