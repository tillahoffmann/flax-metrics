import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal

from flax_metrics import RecallAtK


def sklearn_recall_at_k(scores, relevance, k):
    """Reference implementation of Recall@K."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    relevant_in_top_k = 0
    total_relevant = 0

    for i in range(scores.shape[0]):
        top_k_indices = np.argsort(-scores[i])[:k]
        relevant_in_top_k += relevance[i, top_k_indices].sum()
        total_relevant += relevance[i].sum()

    return relevant_in_top_k / total_relevant


@pytest.mark.parametrize(
    "scores,relevance,k",
    [
        # Perfect recall@2: both relevant items in top 2
        ([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], 2),
        # Zero recall@2: no relevant items in top 2
        ([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0], 2),
        # Partial recall@2: 1 of 2 relevant items in top 2
        ([0.9, 0.1, 0.8, 0.2], [1, 1, 0, 0], 2),
        # Different k value
        ([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], 3),
    ],
)
def test_recall_at_k_matches_reference(scores, relevance, k):
    """Verify RecallAtK matches reference implementation."""
    scores = jnp.array(scores)
    relevance = jnp.array(relevance)

    metric = RecallAtK(k=k)
    metric.update(scores=scores, relevance=relevance)
    actual = float(metric.compute())

    expected = sklearn_recall_at_k(scores, relevance, k)

    assert_almost_equal(actual, expected)


def test_recall_at_k_batched():
    """Verify batched queries work correctly."""
    scores = jnp.array(
        [
            [0.9, 0.8, 0.1, 0.2],
            [0.1, 0.2, 0.9, 0.8],
        ]
    )
    relevance = jnp.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )

    metric = RecallAtK(k=2)
    metric.update(scores=scores, relevance=relevance)
    actual = float(metric.compute())

    expected = sklearn_recall_at_k(scores, relevance, k=2)

    assert_almost_equal(actual, expected)


def test_recall_at_k_accumulation():
    """Accumulated metric over batches matches reference on combined data."""
    metric = RecallAtK(k=2)

    scores1 = jnp.array([0.9, 0.8, 0.1, 0.2])
    relevance1 = jnp.array([1, 1, 0, 0])
    metric.update(scores=scores1, relevance=relevance1)

    scores2 = jnp.array([0.1, 0.2, 0.9, 0.8])
    relevance2 = jnp.array([1, 1, 0, 0])
    metric.update(scores=scores2, relevance=relevance2)

    actual = float(metric.compute())

    all_scores = jnp.stack([scores1, scores2])
    all_relevance = jnp.stack([relevance1, relevance2])
    expected = sklearn_recall_at_k(all_scores, all_relevance, k=2)

    assert_almost_equal(actual, expected)


def test_recall_at_k_reset():
    """Reset clears accumulated state."""
    metric = RecallAtK(k=2)

    metric.update(
        scores=jnp.array([0.9, 0.8, 0.1, 0.2]),
        relevance=jnp.array([1, 1, 0, 0]),
    )
    metric.reset()
    metric.update(
        scores=jnp.array([0.1, 0.2, 0.9, 0.8]),
        relevance=jnp.array([1, 1, 0, 0]),
    )

    actual = float(metric.compute())

    expected = sklearn_recall_at_k([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0], k=2)

    assert_almost_equal(actual, expected)
