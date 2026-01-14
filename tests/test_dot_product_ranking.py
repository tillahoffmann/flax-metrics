"""Tests for DotProduct* ranking metrics.

These tests verify DotProduct* metrics work correctly by testing:
1. Equivalence to ir-measures reference (proves correctness)
2. Unique features: per-query indices, k > subset, variable sizes, multi-batch
"""

import ir_measures
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute
from ir_measures import AP, RR, P, R, nDCG
from numpy.testing import assert_almost_equal

from flax_metrics import (
    DotProductMeanAveragePrecision,
    DotProductMeanReciprocalRank,
    DotProductNDCG,
    DotProductPrecisionAtK,
    DotProductRecallAtK,
)


def ir_measures_score(measure, scores, relevance):
    """Compute a metric using ir-measures as reference implementation."""
    scores = np.asarray(scores)
    relevance = np.asarray(relevance)

    if scores.ndim == 1:
        scores = scores[None, :]
        relevance = relevance[None, :]

    num_items = scores.shape[-1]
    scores = scores.reshape(-1, num_items)
    relevance = relevance.reshape(-1, num_items)
    num_queries = scores.shape[0]

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

    results = ir_measures.calc_aggregate([measure], qrels, run)
    return results[measure]


# Map DotProduct* classes to ir-measures measures
DOT_PRODUCT_METRICS = [
    (DotProductPrecisionAtK, lambda k: P @ k),
    (DotProductRecallAtK, lambda k: R @ k),
    (DotProductMeanReciprocalRank, lambda k: RR @ k),
    (DotProductMeanAveragePrecision, lambda k: AP @ k),
    (DotProductNDCG, lambda k: nDCG @ k),
]


@pytest.mark.parametrize("dp_cls,ir_measure_fn", DOT_PRODUCT_METRICS)
@pytest.mark.parametrize(
    "batch_shape,num_candidates,k",
    [
        ((4,), 5, 2),
        ((2, 3), 5, 2),
        ((2, 2, 2), 5, 2),
        # Larger depth than items available
        ((2,), 4, 10),
    ],
)
def test_dot_product_matches_ir_measures(
    dp_cls, ir_measure_fn, batch_shape, num_candidates, k, jit
):
    """DotProduct* matches ir-measures for various batch shapes."""
    rng = np.random.default_rng(123)
    num_features = 4

    query = rng.standard_normal((*batch_shape, num_features)).astype(np.float32)
    keys = rng.standard_normal((num_candidates, num_features)).astype(np.float32)
    indices = np.tile(np.arange(num_candidates), (*batch_shape, 1))
    relevance = rng.integers(0, 3, size=(*batch_shape, num_candidates)).astype(
        np.float32
    )

    # Compute expected scores for ir-measures comparison
    scores = np.einsum("...f,cf->...c", query, keys)

    dp_metric = dp_cls(k=k)
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    dp_update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        labels=jnp.array(relevance),
    )

    actual = float(dp_compute())
    expected = ir_measures_score(ir_measure_fn(k), scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("dp_cls,ir_measure_fn", DOT_PRODUCT_METRICS)
def test_dot_product_accumulation(dp_cls, ir_measure_fn, jit):
    """DotProduct* accumulation matches ir-measures on combined data."""
    rng = np.random.default_rng(456)
    num_features = 8
    num_candidates = 5
    k = 2

    dp_metric = dp_cls(k=k)
    dp_update, dp_compute = update_and_compute(dp_metric, jit)

    all_scores = []
    all_relevance = []

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
            labels=jnp.array(relevance),
        )

        all_scores.append(scores)
        all_relevance.append(relevance)

    actual = float(dp_compute())
    expected = ir_measures_score(
        ir_measure_fn(k),
        np.concatenate(all_scores, axis=0),
        np.concatenate(all_relevance, axis=0),
    )

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("dp_cls,ir_measure_fn", DOT_PRODUCT_METRICS)
def test_dot_product_per_query_indices(dp_cls, ir_measure_fn, jit):
    """Per-query indices correctly select different items for each query."""
    rng = np.random.default_rng(42)
    num_features = 4
    num_candidates = 10
    batch_size = 3
    num_sampled = 5
    k = 3

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

    dp_metric = dp_cls(k=k)
    dp_update, dp_compute = update_and_compute(dp_metric, jit)
    dp_update(
        query=jnp.array(query),
        keys=jnp.array(keys),
        indices=jnp.array(indices),
        labels=jnp.array(relevance),
    )

    actual = float(dp_compute())
    expected = ir_measures_score(ir_measure_fn(k), expected_scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("dp_cls,ir_measure_fn", DOT_PRODUCT_METRICS)
def test_dot_product_k_larger_than_subset(dp_cls, ir_measure_fn, jit):
    """When k > num_sampled, metrics match ir-measures at declared k."""
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
        labels=jnp.array(relevance),
    )

    actual = float(dp_compute())
    # Compare with ir-measures at declared k (not effective_k)
    expected = ir_measures_score(ir_measure_fn(10), scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


def test_dot_product_variable_subset_sizes(jit):
    """Variable subset sizes across updates are handled correctly.

    Tests that DotProductPrecisionAtK uses declared k in denominator,
    matching ir-measures behavior.
    """
    # Use one-hot keys so scores = query values directly
    dp_metric = DotProductPrecisionAtK(k=3)
    update, compute = update_and_compute(dp_metric, jit)

    # First update: 4 items, top-3 considered
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
        labels=jnp.array(relevance1),
    )

    # Second update: 2 items (k=3 but only 2 available)
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
        labels=jnp.array(relevance2),
    )

    result = float(compute())
    # Total relevant in top-k: 2 (from first) + 1 (from second) = 3
    # Total items considered: 3 + 3 = 6 (uses declared k, not effective_k)
    # Expected precision: 3/6 = 0.5
    assert_almost_equal(result, 0.5, decimal=5)
