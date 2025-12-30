import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import ndcg_score

from flax_metrics import (
    NDCG,
    MeanAveragePrecision,
    MeanReciprocalRank,
    PrecisionAtK,
    RecallAtK,
)


def precision_at_k(scores, relevance, k):
    """Reference implementation of Precision@K."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    relevant_in_top_k = 0
    num_queries = scores.shape[0]

    for i in range(num_queries):
        top_k_indices = np.argsort(-scores[i])[:k]
        relevant_in_top_k += relevance[i, top_k_indices].sum()

    return relevant_in_top_k / (num_queries * k)


def recall_at_k(scores, relevance, k):
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


def mrr(scores, relevance, k):
    """Reference implementation of Mean Reciprocal Rank."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    total_rr = 0.0
    num_queries = scores.shape[0]

    for i in range(num_queries):
        top_k_indices = np.argsort(-scores[i])[:k]
        top_k_rel = relevance[i, top_k_indices]
        # Find first relevant item
        relevant_positions = np.where(top_k_rel > 0)[0]
        if len(relevant_positions) > 0:
            total_rr += 1.0 / (relevant_positions[0] + 1)

    return total_rr / num_queries


def mean_average_precision(scores, relevance, k):
    """Reference implementation of Mean Average Precision."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    total_ap = 0.0
    num_queries = scores.shape[0]

    for i in range(num_queries):
        top_k_indices = np.argsort(-scores[i])[:k]
        top_k_rel = relevance[i, top_k_indices]

        # Compute AP (using binary relevance)
        total_relevant = (relevance[i] > 0).sum()
        if total_relevant == 0:
            continue

        cumsum = 0
        ap_sum = 0.0
        for pos, rel in enumerate(top_k_rel):
            if rel > 0:
                cumsum += 1
                precision_at_pos = cumsum / (pos + 1)
                ap_sum += precision_at_pos

        total_ap += ap_sum / total_relevant

    return total_ap / num_queries


def sklearn_ndcg(scores, relevance, k):
    """Wrapper for sklearn's ndcg_score."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    return ndcg_score(relevance, scores, k=k)


METRICS = [
    (PrecisionAtK, precision_at_k),
    (RecallAtK, recall_at_k),
    (MeanReciprocalRank, mrr),
    (MeanAveragePrecision, mean_average_precision),
    (NDCG, sklearn_ndcg),
]


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
@pytest.mark.parametrize(
    "scores,relevance,k",
    [
        # Perfect ranking
        ([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], 2),
        # Worst ranking
        ([0.1, 0.2, 0.9, 0.8], [1, 1, 0, 0], 2),
        # Partial ranking
        ([0.9, 0.1, 0.8, 0.2], [1, 1, 0, 0], 2),
        # Different k value
        ([0.9, 0.8, 0.7, 0.6], [1, 0, 1, 0], 3),
        # Graded relevance
        ([0.9, 0.8, 0.7, 0.6], [3, 2, 1, 0], 3),
    ],
)
def test_metric_matches_sklearn(metric_cls, sklearn_fn, scores, relevance, k):
    """Verify our metrics match sklearn."""
    scores = jnp.array(scores, dtype=jnp.float32)
    relevance = jnp.array(relevance, dtype=jnp.float32)

    metric = metric_cls(k=k)
    metric.update(scores=scores, relevance=relevance)
    actual = float(metric.compute())

    expected = sklearn_fn(scores, relevance, k)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_batched(metric_cls, sklearn_fn):
    """Verify batched queries work correctly."""
    scores = jnp.array(
        [
            [0.9, 0.8, 0.1, 0.2],
            [0.1, 0.2, 0.9, 0.8],
        ],
        dtype=jnp.float32,
    )
    relevance = jnp.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=jnp.float32,
    )

    metric = metric_cls(k=2)
    metric.update(scores=scores, relevance=relevance)
    actual = float(metric.compute())

    expected = sklearn_fn(scores, relevance, k=2)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_accumulation(metric_cls, sklearn_fn):
    """Accumulated metric over batches matches sklearn on combined data."""
    metric = metric_cls(k=2)

    scores1 = jnp.array([0.9, 0.8, 0.1, 0.2], dtype=jnp.float32)
    relevance1 = jnp.array([1, 1, 0, 0], dtype=jnp.float32)
    metric.update(scores=scores1, relevance=relevance1)

    scores2 = jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32)
    relevance2 = jnp.array([1, 1, 0, 0], dtype=jnp.float32)
    metric.update(scores=scores2, relevance=relevance2)

    actual = float(metric.compute())

    all_scores = jnp.stack([scores1, scores2])
    all_relevance = jnp.stack([relevance1, relevance2])
    expected = sklearn_fn(all_scores, all_relevance, k=2)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,sklearn_fn", METRICS)
def test_metric_reset(metric_cls, sklearn_fn):
    """Reset clears accumulated state."""
    metric = metric_cls(k=2)

    metric.update(
        scores=jnp.array([0.9, 0.8, 0.1, 0.2], dtype=jnp.float32),
        relevance=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )
    metric.reset()
    metric.update(
        scores=jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32),
        relevance=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )

    actual = float(metric.compute())

    expected = sklearn_fn(
        np.array([[0.1, 0.2, 0.9, 0.8]]),
        np.array([[1, 1, 0, 0]]),
        k=2,
    )

    assert_almost_equal(actual, expected, decimal=5)
