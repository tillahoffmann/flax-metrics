import jax.numpy as jnp
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import recall_score

from flax_metrics import Recall


@pytest.mark.parametrize(
    "logits,labels,threshold",
    [
        # Perfect recall
        ([1.0, 1.0, -1.0, -1.0], [1, 1, 0, 0], 0.0),
        # Zero recall
        ([-1.0, -1.0, 1.0, 1.0], [1, 1, 0, 0], 0.0),
        # Partial recall
        ([1.0, -1.0, -1.0, 1.0], [1, 1, 0, 0], 0.0),
        # Custom threshold
        ([0.6, 0.4, 0.3, 0.8], [1, 1, 0, 0], 0.5),
        # Threshold boundary (strict inequality)
        ([0.5, 0.5], [1, 1], 0.5),
    ],
)
def test_recall_matches_sklearn(logits, labels, threshold):
    """Verify our recall matches sklearn's recall_score."""
    logits = jnp.array(logits)
    labels = jnp.array(labels)

    metric = Recall(threshold=threshold)
    metric.update(logits=logits, labels=labels)
    actual = float(metric.compute())

    predictions = (logits > threshold).astype(int)
    expected = recall_score(labels, predictions)

    assert_almost_equal(actual, expected)


def test_recall_accumulation_matches_sklearn():
    """Accumulated recall over batches matches sklearn on combined data."""
    metric = Recall(threshold=0.0)

    logits1 = jnp.array([1.0, -1.0])
    labels1 = jnp.array([1, 0])
    metric.update(logits=logits1, labels=labels1)

    logits2 = jnp.array([1.0, -1.0, -1.0])
    labels2 = jnp.array([1, 1, 0])
    metric.update(logits=logits2, labels=labels2)

    actual = float(metric.compute())

    # Compare against sklearn on combined data
    all_logits = jnp.concatenate([logits1, logits2])
    all_labels = jnp.concatenate([labels1, labels2])
    predictions = (all_logits > 0.0).astype(int)
    expected = recall_score(all_labels, predictions)

    assert_almost_equal(actual, expected)


def test_recall_reset():
    """Reset clears accumulated state."""
    metric = Recall(threshold=0.0)

    metric.update(logits=jnp.array([1.0, 1.0]), labels=jnp.array([1, 1]))
    metric.reset()
    metric.update(logits=jnp.array([-1.0, -1.0]), labels=jnp.array([1, 1]))

    actual = float(metric.compute())

    predictions = jnp.array([0, 0])
    expected = recall_score([1, 1], predictions)

    assert_almost_equal(actual, expected)
