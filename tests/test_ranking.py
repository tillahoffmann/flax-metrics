import ir_measures
import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from ir_measures import AP, RR, P, R, nDCG
from numpy.testing import assert_almost_equal

from flax_metrics import (
    NDCG,
    MeanAveragePrecision,
    MeanReciprocalRank,
    PrecisionAtK,
    RecallAtK,
)


def ir_measures_score(measure, scores, labels):
    """Compute a metric using ir-measures as reference implementation.

    Converts scores/relevance arrays to ir-measures format and computes the metric.

    Args:
        measure: An ir-measures measure (e.g., P@2, nDCG@3, RR, AP).
        scores: Array of shape (..., num_items) with predicted scores.
        labels: Array of shape (..., num_items) with relevance labels.

    Returns:
        The computed metric value averaged across all queries.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)

    if scores.ndim == 1:
        scores = scores[None, :]
        labels = labels[None, :]

    # Flatten batch dimensions to (num_queries, num_items)
    num_items = scores.shape[-1]
    scores = scores.reshape(-1, num_items)
    labels = labels.reshape(-1, num_items)
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
            qrels[q_id][d_id] = int(labels[q_idx, d_idx])
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
        # Larger depth than items available.
        ([0.9, 0.8, 0.7, 0.6], [1, 3, 2, 0], 10),
    ],
)
def test_metric_matches_ir_measures(
    metric_cls, ir_measure_fn, scores, relevance, k, jit, masked
):
    """Verify our metrics match ir-measures."""
    scores = jnp.array(scores, dtype=jnp.float32)
    relevance = jnp.array(relevance, dtype=jnp.float32)

    metric = metric_cls(k=k)
    if masked:
        validate_masking(
            metric, (), {"scores": scores, "labels": relevance}, jit=jit, event_dim=1
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(scores=scores, labels=relevance)
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
    update(scores=scores1, labels=relevance1)

    scores2 = jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32)
    relevance2 = jnp.array([1, 1, 0, 0], dtype=jnp.float32)
    update(scores=scores2, labels=relevance2)

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
        labels=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )
    metric.reset()
    update(
        scores=jnp.array([0.1, 0.2, 0.9, 0.8], dtype=jnp.float32),
        labels=jnp.array([1, 1, 0, 0], dtype=jnp.float32),
    )

    actual = float(compute())

    expected = ir_measures_score(
        ir_measure_fn(k),
        np.array([[0.1, 0.2, 0.9, 0.8]]),
        np.array([[1, 1, 0, 0]]),
    )

    assert_almost_equal(actual, expected, decimal=5)
