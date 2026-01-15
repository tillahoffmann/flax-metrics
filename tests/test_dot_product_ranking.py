"""Tests for DotProduct* ranking metrics.

These tests verify DotProduct* metrics work correctly by testing:
1. Equivalence to ir-measures reference (proves correctness)
2. Unique features: per-query indices, k > subset, variable sizes, multi-batch
"""

import ir_measures
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from ir_measures import AP, RR, P, R, nDCG
from jax import random
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
METRICS = [
    (DotProductPrecisionAtK, lambda k: P @ k),
    (DotProductRecallAtK, lambda k: R @ k),
    (DotProductMeanReciprocalRank, lambda k: RR @ k),
    (DotProductMeanAveragePrecision, lambda k: AP @ k),
    (DotProductNDCG, lambda k: nDCG @ k),
]


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
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
    metric_cls, ir_measure_fn, batch_shape, num_candidates, k, jit, masked
):
    """DotProduct* matches ir-measures for various batch shapes."""
    key = random.key(123)
    k1, k2, k3 = random.split(key, 3)
    num_features = 4

    query = random.normal(k1, (*batch_shape, num_features))
    keys = random.normal(k2, (num_candidates, num_features))
    indices = jnp.tile(jnp.arange(num_candidates), (*batch_shape, 1))
    relevance = random.randint(k3, (*batch_shape, num_candidates), 0, 3).astype(
        jnp.float32
    )

    metric = metric_cls(k=k)
    if masked:
        # keys is shared across queries and should NOT be masked.
        validate_masking(
            metric,
            (),
            {"query": query, "indices": indices, "labels": relevance},
            jit=jit,
            event_dim=1,
            static_kwargs={"keys": keys},
        )
        return

    # Compute expected scores for ir-measures comparison
    scores = np.einsum("...f,cf->...c", query, keys)

    update, compute = update_and_compute(metric, jit)
    update(
        query=query,
        keys=keys,
        indices=indices,
        labels=relevance,
    )

    actual = float(compute())
    expected = ir_measures_score(ir_measure_fn(k), scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
def test_dot_product_accumulation(metric_cls, ir_measure_fn, jit):
    """DotProduct* accumulation matches ir-measures on combined data."""
    key = random.key(456)
    num_features = 8
    num_candidates = 5
    k = 2

    metric = metric_cls(k=k)
    update, compute = update_and_compute(metric, jit)

    all_scores = []
    all_relevance = []

    for batch_size in [2, 3, 1]:
        key, k1, k2, k3 = random.split(key, 4)
        query = random.normal(k1, (batch_size, num_features))
        keys = random.normal(k2, (num_candidates, num_features))
        indices = jnp.tile(jnp.arange(num_candidates), (batch_size, 1))
        relevance = random.randint(k3, (batch_size, num_candidates), 0, 2).astype(
            jnp.float32
        )
        scores = query @ keys.T

        update(
            query=query,
            keys=keys,
            indices=indices,
            labels=relevance,
        )

        all_scores.append(scores)
        all_relevance.append(relevance)

    actual = float(compute())
    expected = ir_measures_score(
        ir_measure_fn(k),
        jnp.concatenate(all_scores, axis=0),
        jnp.concatenate(all_relevance, axis=0),
    )

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
def test_dot_product_per_query_indices(metric_cls, ir_measure_fn, jit):
    """Per-query indices correctly select different items for each query."""
    key = random.key(42)
    k1, k2 = random.split(key)
    num_features = 4
    num_candidates = 10
    batch_size = 3
    num_sampled = 5
    k = 3

    keys = random.normal(k1, (num_candidates, num_features))
    query = random.normal(k2, (batch_size, num_features))
    indices = jnp.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [0, 2, 4, 6, 8]])
    relevance = jnp.array(
        [[1, 0, 1, 0, 0], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1]], dtype=jnp.float32
    )

    # Compute expected scores
    expected_scores = jnp.zeros((batch_size, num_sampled), dtype=jnp.float32)
    for b in range(batch_size):
        for s in range(num_sampled):
            expected_scores = expected_scores.at[b, s].set(
                query[b] @ keys[indices[b, s]]
            )

    metric = metric_cls(k=k)
    update, compute = update_and_compute(metric, jit)
    update(
        query=query,
        keys=keys,
        indices=indices,
        labels=relevance,
    )

    actual = float(compute())
    expected = ir_measures_score(ir_measure_fn(k), expected_scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize("metric_cls,ir_measure_fn", METRICS)
def test_dot_product_k_larger_than_subset(metric_cls, ir_measure_fn, jit):
    """When k > num_sampled, metrics match ir-measures at declared k."""
    key = random.key(111)
    k1, k2 = random.split(key)
    num_features = 4
    num_candidates = 4  # Only 4 items, but k=10

    query = random.normal(k1, (2, num_features))
    keys = random.normal(k2, (num_candidates, num_features))
    indices = jnp.tile(jnp.arange(num_candidates), (2, 1))
    relevance = jnp.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=jnp.float32)
    scores = query @ keys.T

    metric = metric_cls(k=10)  # k > num_candidates
    update, compute = update_and_compute(metric, jit)
    update(query=query, keys=keys, indices=indices, labels=relevance)

    actual = float(compute())
    # Compare with ir-measures at declared k (not effective_k)
    expected = ir_measures_score(ir_measure_fn(10), scores, relevance)

    assert_almost_equal(actual, expected, decimal=5)


def test_dot_product_variable_subset_sizes(jit):
    """Variable subset sizes across updates are handled correctly.

    Tests that DotProductPrecisionAtK uses declared k in denominator,
    matching ir-measures behavior.
    """
    # Use one-hot keys so scores = query values directly
    metric = DotProductPrecisionAtK(k=3)
    update, compute = update_and_compute(metric, jit)

    # First update: 4 items, top-3 considered
    # Scores will be [0.9, 0.8, 0.7, 0.6], top-3 are indices 0,1,2
    # Relevance [1, 1, 0, 0] -> 2 relevant in top-3
    update(
        query=jnp.array([[0.9, 0.8, 0.7, 0.6]]),
        keys=jnp.eye(4),
        indices=jnp.arange(4).reshape(1, -1),
        labels=jnp.array([[1, 1, 0, 0]], dtype=jnp.float32),
    )

    # Second update: 2 items (k=3 but only 2 available)
    # Scores will be [0.9, 0.1], top-2 are indices 0,1
    # Relevance [1, 0] -> 1 relevant in top-2
    update(
        query=jnp.array([[0.9, 0.1]]),
        keys=jnp.eye(2),
        indices=jnp.arange(2).reshape(1, -1),
        labels=jnp.array([[1, 0]], dtype=jnp.float32),
    )

    result = float(compute())
    # Total relevant in top-k: 2 (from first) + 1 (from second) = 3
    # Total items considered: 3 + 3 = 6 (uses declared k, not effective_k)
    # Expected precision: 3/6 = 0.5
    assert_almost_equal(result, 0.5, decimal=5)
