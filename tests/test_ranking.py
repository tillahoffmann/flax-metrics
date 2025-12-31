import ir_measures
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute
from ir_measures import AP, RR, P, R, nDCG
from numpy.testing import assert_almost_equal

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


def ir_measures_score(measure, scores, relevance):
    """Compute a metric using ir-measures as reference implementation.

    Converts scores/relevance arrays to ir-measures format and computes the metric.

    Args:
        measure: An ir-measures measure (e.g., P@2, nDCG@3, RR, AP).
        scores: Array of shape (..., num_items) with predicted scores.
        relevance: Array of shape (..., num_items) with relevance labels.

    Returns:
        The computed metric value averaged across all queries.
    """
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    # Flatten batch dimensions to (num_queries, num_items)
    num_items = scores.shape[-1]
    scores = scores.reshape(-1, num_items)
    relevance = relevance.reshape(-1, num_items)
    num_queries = scores.shape[0]

    # Convert to ir-measures format
    qrels = {}
    run = {}
    for q_idx in range(num_queries):
        q_id = f"q{q_idx}"
        qrels[q_id] = {}
        run[q_id] = {}
        for d_idx in range(num_items):
            d_id = f"d{d_idx}"
            qrels[q_id][d_id] = int(relevance[q_idx, d_idx])
            run[q_id][d_id] = float(scores[q_idx, d_idx])

    # Compute metric
    results = ir_measures.calc_aggregate([measure], qrels, run)
    return results[measure]


# Map flax-metrics classes to ir-measures measures (k is filled in at test time)
METRICS = [
    (PrecisionAtK, lambda k: P @ k),
    (RecallAtK, lambda k: R @ k),
    (MeanReciprocalRank, lambda k: RR @ k),
    (MeanAveragePrecision, lambda k: AP @ k),
    (NDCG, lambda k: nDCG @ k),
]


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
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
def test_metric_matches_ir_measures(
    metric_cls, ir_measure_fn, scores, relevance, k, jit
):
    """Verify our metrics match ir-measures."""
    scores = jnp.array(scores, dtype=jnp.float32)
    relevance = jnp.array(relevance, dtype=jnp.float32)

    metric = metric_cls(k=k)
    update, compute = update_and_compute(metric, jit)
    update(scores=scores, relevance=relevance)
    actual = float(compute())

    expected = ir_measures_score(ir_measure_fn(k), scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
def test_metric_accumulation(metric_cls, ir_measure_fn, jit):
    """Accumulated metric over batches matches ir-measures on combined data."""
    k = 2
    metric = metric_cls(k=k)
    update, compute = update_and_compute(metric, jit)

    scores1 = jnp.array([0.9, 0.8, 0.1, 0.2], dtype=jnp.float32)
    relevance1 = jnp.array([1, 1, 0, 0], dtype=jnp.float32)
    update(scores=scores1, relevance=relevance1)

    scores2 = jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32)
    relevance2 = jnp.array([1, 1, 0, 0], dtype=jnp.float32)
    update(scores=scores2, relevance=relevance2)

    actual = float(compute())

    all_scores = jnp.stack([scores1, scores2])
    all_relevance = jnp.stack([relevance1, relevance2])
    expected = ir_measures_score(ir_measure_fn(k), all_scores, all_relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
def test_metric_reset(metric_cls, ir_measure_fn, jit):
    """Reset clears accumulated state."""
    k = 2
    metric = metric_cls(k=k)
    update, compute = update_and_compute(metric, jit)

    update(
        scores=jnp.array([0.9, 0.8, 0.1, 0.2], dtype=jnp.float32),
        relevance=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )
    metric.reset()
    update(
        scores=jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32),
        relevance=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )

    actual = float(compute())

    expected = ir_measures_score(
        ir_measure_fn(k),
        np.array([[0.1, 0.2, 0.9, 0.8]]),
        np.array([[1, 1, 0, 0]]),
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
def test_dot_product_equals_base(dp_cls, base_cls, batch_shape, jit):
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
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    dp_update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    base_metric = base_cls(k=k)
    base_update, base_compute = update_and_compute(base_metric, jit)
    base_update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(float(dp_compute()), float(base_compute()), decimal=5)


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_accumulation(dp_cls, base_cls, jit):
    """DotProduct* accumulation matches base metric accumulation."""
    rng = np.random.default_rng(456)
    num_features = 8
    num_candidates = 5
    k = 2

    dp_metric = dp_cls(k=k)
    base_metric = base_cls(k=k)
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    base_update, base_compute = update_and_compute(base_metric, jit)

    for batch_size in [2, 3, 1]:
        query = rng.standard_normal((batch_size, num_features)).astype(np.float32)
        keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
        indices = np.tile(np.arange(num_candidates), (batch_size, 1))
        relevance = rng.integers(0, 2, size=(batch_size, num_candidates)).astype(
            np.float32
        )
        scores = query @ keys.T

        dp_update(
            query=jnp.array(query),
            keys=jnp.array(keys),
            indices=jnp.array(indices),
            relevance=jnp.array(relevance),
        )
        base_update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(float(dp_compute()), float(base_compute()), decimal=5)


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_per_query_indices(dp_cls, base_cls, jit):
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
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    dp_update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    base_metric = base_cls(k=3)
    base_update, base_compute = update_and_compute(base_metric, jit)
    base_update(scores=jnp.array(expected_scores), relevance=jnp.array(relevance))

    assert_almost_equal(float(dp_compute()), float(base_compute()), decimal=5)


@pytest.mark.parametrize("dp_cls,base_cls", DOT_PRODUCT_TO_BASE)
def test_dot_product_k_larger_than_subset(dp_cls, base_cls, jit):
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
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    dp_update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        relevance=jnp.array(relevance),
    )

    # Should match base metric with k=4 (effective_k)
    base_metric = base_cls(k=4)
    base_update, base_compute = update_and_compute(base_metric, jit)
    base_update(scores=jnp.array(scores), relevance=jnp.array(relevance))

    assert_almost_equal(float(dp_compute()), float(base_compute()), decimal=5)


def test_dot_product_variable_subset_sizes(jit):
    """Variable subset sizes across updates are handled correctly.

    Tests that DotProductPrecisionAtK correctly tracks total_items_considered
    (sum of effective_k) rather than num_queries * k.
    """
    # Use one-hot keys so scores = query values directly
    dp_metric = DotProductPrecisionAtK(k=3)
    update, compute = update_and_compute(dp_metric, jit)

    # First update: 4 items, effective_k=3
    # Scores will be [0.9, 0.8, 0.7, 0.6], top-3 are indices 0,1,2
    # Relevance [1, 1, 0, 0] -> 2 relevant in top-3
    keys1 = np.eye(4, dtype=np.float32)
    query1 = np.array([[0.9, 0.8, 0.7, 0.6]], dtype=np.float32)
    indices1 = np.arange(4).reshape(1, -1)
    relevance1 = np.array([[1, 1, 0, 0]], dtype=np.float32)
    update(
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
    update(
        query=jnp.array(query2),
        keys=jnp.array(keys2),
        indices=jnp.array(indices2),
        relevance=jnp.array(relevance2),
    )

    result = float(compute())
    # Total relevant in top-k: 2 (from first) + 1 (from second) = 3
    # Total items considered: 3 + 2 = 5
    # Expected precision: 3/5 = 0.6
    assert_almost_equal(result, 0.6, decimal=5)
