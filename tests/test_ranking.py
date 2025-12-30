import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn.metrics import ndcg_score

from flax_metrics import (
    NDCG,
    DotProductMeanAveragePrecision,
    DotProductMeanReciprocalRank,
    DotProductNDCG,
    DotProductPrecisionAtK,
    DotProductRecallAtK,
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
        # Graded relevance where highest relevance is not first (tests MRR correctly
        # finds first relevant item, not highest relevance item)
        ([0.9, 0.8, 0.7, 0.6], [1, 3, 2, 0], 3),
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


# --- DotProduct* metric tests ---
# These tests verify DotProduct* metrics work correctly.
# Base metric correctness is tested above; here we test:
# 1. Equivalence to base metrics (proves correctness)
# 2. Unique features: per-query indices, k > subset, variable sizes, multi-batch

DOT_PRODUCT_TO_BASE = [
    (DotProductPrecisionAtK, PrecisionAtK),
    (DotProductRecallAtK, RecallAtK),
    (DotProductMeanReciprocalRank, MeanReciprocalRank),
    (DotProductMeanAveragePrecision, MeanAveragePrecision),
    (DotProductNDCG, NDCG),
]


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
@pytest.mark.parametrize("batch_shape", [(4,), (2, 3), (2, 2, 2)])
def test_dot_product_equals_base(dp_cls, base_cls, batch_shape):
    """DotProduct* equals base metric for various batch shapes."""
    rng = np.random.default_rng(123)
    num_features = 4
    num_candidates = 5
    k = 2

    query = rng.standard_normal((*batch_shape, num_features)).astype(np.float32)
    keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
    indices = np.tile(np.arange(num_candidates), (*batch_shape, 1))
    relevance = rng.integers(0, 3, size=(*batch_shape, num_candidates)).astype(
        np.float32
    )
    scores = np.einsum("...f,cf->...c", query, keys)

    dp_metric = dp_cls(k=k)
    dp_metric.update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    base_metric = base_cls(k=k)
    base_metric.update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(
        float(dp_metric.compute()), float(base_metric.compute()), decimal=5
    )


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_accumulation(dp_cls, base_cls):
    """DotProduct* accumulation matches base metric accumulation."""
    rng = np.random.default_rng(456)
    num_features = 8
    num_candidates = 5
    k = 2

    dp_metric = dp_cls(k=k)
    base_metric = base_cls(k=k)

    for batch_size in [2, 3, 1]:
        query = rng.standard_normal((batch_size, num_features)).astype(np.float32)
        keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
        indices = np.tile(np.arange(num_candidates), (batch_size, 1))
        relevance = rng.integers(0, 2, size=(batch_size, num_candidates)).astype(
            np.float32
        )
        scores = query @ keys.T

        dp_metric.update(
            query=jnp.array(query),
            keys=jnp.array(keys),
            indices=jnp.array(indices),
            relevance=jnp.array(relevance),
        )
        base_metric.update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(
        float(dp_metric.compute()), float(base_metric.compute()), decimal=5
    )


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_per_query_indices(dp_cls, base_cls):
    """Per-query indices correctly select different items for each query."""
    rng = np.random.default_rng(42)
    num_features = 4
    num_candidates = 10
    batch_size = 3
    num_sampled = 5

    keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
    query = rng.standard_normal((batch_size, num_features)).astype(np.float32)
    indices = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 2, 4, 6, 8]])
    relevance = np.array(
        [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]], dtype=np.float32
    )

    # Compute expected scores
    expected_scores = np.zeros((batch_size, num_sampled), dtype=np.float32)
    for b in range(batch_size):
        for s in range(num_sampled):
            expected_scores[b, s] = query[b] @ keys[indices[b, s]]

    dp_metric = dp_cls(k=3)
    dp_metric.update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    base_metric = base_cls(k=3)
    base_metric.update(
        scores=jnp.array(expected_scores), relevance=jnp.array(relevance)
    )

    assert_almost_equal(
        float(dp_metric.compute()), float(base_metric.compute()), decimal=5
    )


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_k_larger_than_subset(dp_cls, base_cls):
    """When k > num_sampled, effective_k = num_sampled is used."""
    rng = np.random.default_rng(111)
    num_features = 4
    num_candidates = 4  # Only 4 items, but k=10

    query = rng.standard_normal((2, num_features)).astype(np.float32)
    keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
    indices = np.tile(np.arange(num_candidates), (2, 1))
    relevance = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=np.float32)
    scores = query @ keys.T

    dp_metric = dp_cls(k=10)  # k > num_candidates
    dp_metric.update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    # Should match base metric with k=4 (effective_k)
    base_metric = base_cls(k=4)
    base_metric.update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(
        float(dp_metric.compute()), float(base_metric.compute()), decimal=5
    )


def test_dot_product_variable_subset_sizes():
    """Variable subset sizes across updates are handled correctly.

    Tests that DotProductPrecisionAtK correctly tracks total_items_considered
    (sum of effective_k) rather than num_queries * k.
    """
    # Use one-hot keys so scores = query values directly
    dp_metric = DotProductPrecisionAtK(k=3)

    # First update: 4 items, effective_k=3
    # Scores will be [0.9, 0.8, 0.7, 0.6], top-3 are indices 0,1,2
    # Relevance [1, 1, 0, 0] -> 2 relevant in top-3
    keys1 = np.eye(4, dtype=np.float32)
    query1 = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    indices1 = np.arange(4).reshape(1, -1)
    relevance1 = np.array([[1, 1, 0, 0]], dtype=np.float32)
    dp_metric.update(
        query=jnp.array(query1),
        keys=jnp.array(keys1),
        indices=jnp.array(indices1),
        relevance=jnp.array(relevance1),
    )

    # Second update: 2 items, effective_k=2 (k=3 but only 2 items)
    # Scores will be [0.9, 0.1], top-2 are indices 0,1
    # Relevance [1, 0] -> 1 relevant in top-2
    keys2 = np.eye(2, dtype=np.float32)
    query2 = np.array([[0.9, 0.1]], dtype=np.float32)
    indices2 = np.arange(2).reshape(1, -1)
    relevance2 = np.array([[1, 0]], dtype=np.float32)
    dp_metric.update(
        query=jnp.array(query2),
        keys=jnp.array(keys2),
        indices=jnp.array(indices2),
        relevance=jnp.array(relevance2),
    )

    result = float(dp_metric.compute())
    # Total relevant in top-k: 2 (from first) + 1 (from second) = 3
    # Total items considered: 3 + 2 = 5
    # Expected precision: 3/5 = 0.6
    assert_almost_equal(result, 0.6, decimal=5)
