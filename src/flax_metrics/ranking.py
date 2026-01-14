"""Metrics for evaluating ranked retrieval results, where items are ranked by
precomputed scores. These metrics compare the ranking against relevance labels to
measure retrieval quality.
"""

from flax import nnx
from jax import lax
from jax import numpy as jnp


class PrecisionAtK(nnx.Metric):
    """Precision@K, the fraction of top-k items that are relevant.

    Args:
        k: Number of top items to consider.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import PrecisionAtK
        >>>
        >>> scores    = jnp.array([0.1, 0.4, 0.3, 0.2])
        >>> relevance = jnp.array([  0,   1,   1,   0])
        >>> metric = PrecisionAtK(k=2)
        >>> metric.update(labels=relevance, scores=scores)
        >>> metric.compute()  # top-2 are indices 1, 2 both relevant
        Array(1., dtype=float32)
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, labels: jnp.ndarray, scores: jnp.ndarray, **_) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the precision@k with a batch of scored items.

        Args:
            labels: Relevance labels, shape :code:`(..., num_items)`.
            scores: Scores for each item, same shape as labels.
        """
        # Flatten batch dimensions to count queries
        num_queries = scores.size // scores.shape[-1]

        # Cap k at the number of items available
        k = min(self.k, scores.shape[-1])

        # Get top-k indices along last axis (descending order)
        _, top_k_indices = lax.top_k(scores, k)

        # Gather relevance values for top-k items
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Accumulate counts (binary relevance: any value > 0 is relevant)
        self.relevant_in_top_k[...] += (top_k_relevance > 0).sum()
        self.num_queries[...] += num_queries

    def compute(self) -> jnp.ndarray:
        """Compute and return the precision@k."""
        return self.relevant_in_top_k[...] / (self.num_queries[...] * self.k)


class RecallAtK(nnx.Metric):
    """Recall@K, the fraction of relevant items that appear in the top-k ranked results.

    Computes mean recall over all queries (macro-average).

    Args:
        k: Number of top items to consider.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import RecallAtK
        >>>
        >>> scores    = jnp.array([0.1, 0.4, 0.3, 0.2])
        >>> relevance = jnp.array([  1,   1,   1,   0])
        >>> metric = RecallAtK(k=2)
        >>> metric.update(labels=relevance, scores=scores)
        >>> metric.compute()  # 2 of 3 relevant items in top-2
        Array(0.6666667, dtype=float32)
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.total_recall = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.total_recall = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, labels: jnp.ndarray, scores: jnp.ndarray, **_) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the recall@k with a batch of scored items.

        Args:
            labels: Relevance labels, shape :code:`(..., num_items)`.
            scores: Scores for each item, same shape as labels.
        """
        # Flatten batch dimensions to (num_queries, num_items)
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        labels = labels.reshape(-1, original_shape[-1])

        # Cap k at the number of items available
        k = min(self.k, scores.shape[-1])

        # Get top-k indices along last axis (descending order)
        _, top_k_indices = lax.top_k(scores, k)

        # Gather relevance values for top-k items
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Compute per-query recall (binary relevance: any value > 0 is relevant)
        relevant_in_top_k = (top_k_relevance > 0).sum(axis=-1)
        total_relevant = (labels > 0).sum(axis=-1)

        # Handle queries with no relevant items (avoid division by zero)
        recall_per_query = jnp.where(
            total_relevant > 0, relevant_in_top_k / total_relevant, 0.0
        )

        self.total_recall[...] += recall_per_query.sum()
        self.num_queries[...] += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        """Compute and return the recall@k."""
        return self.total_recall[...] / self.num_queries[...]


class MeanReciprocalRank(nnx.Metric):
    """Mean Reciprocal Rank.

    The average of reciprocal ranks of the first relevant item for each query.

    Args:
        k: Number of top items to consider. If None, considers all items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import MeanReciprocalRank
        >>>
        >>> scores    = jnp.array([0.1, 0.4, 0.3, 0.2])
        >>> relevance = jnp.array([  1,   0,   0,   1])
        >>> metric = MeanReciprocalRank()
        >>> metric.update(labels=relevance, scores=scores)
        >>> metric.compute()  # first relevant at rank 3
        Array(0.33333334, dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, labels: jnp.ndarray, scores: jnp.ndarray, **_) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the mean reciprocal rank with a batch of scored items.

        Args:
            labels: Relevance labels, shape :code:`(..., num_items)`.
            scores: Scores for each item, same shape as labels.
        """
        # Flatten batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        labels = labels.reshape(-1, original_shape[-1])

        # Cap k at the number of items available
        k = min(self.k, scores.shape[-1]) if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Find rank of first relevant item (1-indexed)
        is_relevant = top_k_relevance > 0
        first_relevant_idx = jnp.argmax(is_relevant, axis=-1)
        has_relevant = jnp.any(is_relevant, axis=-1)

        # Reciprocal rank: 1/(rank), where rank = index + 1
        reciprocal_rank = jnp.where(
            has_relevant,
            1.0 / (first_relevant_idx + 1),
            0.0,
        )

        self.total_rr[...] += reciprocal_rank.sum()
        self.num_queries[...] += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        """Compute and return the mean reciprocal rank."""
        return self.total_rr[...] / self.num_queries[...]


class MeanAveragePrecision(nnx.Metric):
    """Mean Average Precision.

    The mean of average precision scores across queries, where average precision
    is the sum of precision@k * rel(k) divided by total relevant items.

    Args:
        k: Number of top items to consider. If None, considers all items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import MeanAveragePrecision
        >>>
        >>> scores    = jnp.array([0.4, 0.3, 0.2, 0.1])
        >>> relevance = jnp.array([  1,   1,   0,   1])
        >>> metric = MeanAveragePrecision()
        >>> metric.update(labels=relevance, scores=scores)
        >>> metric.compute()  # (1/1 + 2/2 + 3/4) / 3
        Array(0.9166667, dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, labels: jnp.ndarray, scores: jnp.ndarray, **_) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the mean average precision with a batch of scored items.

        Args:
            labels: Relevance labels, shape :code:`(..., num_items)`.
            scores: Scores for each item, same shape as labels.
        """
        # Flatten batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        labels = labels.reshape(-1, original_shape[-1])

        # Cap k at the number of items available
        k = min(self.k, scores.shape[-1]) if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Convert to binary relevance for MAP
        top_k_binary = (top_k_relevance > 0).astype(jnp.float32)

        # Compute cumulative sum of relevant items at each position
        cumsum_rel = jnp.cumsum(top_k_binary, axis=-1)

        # Precision at each position: cumsum_rel / position
        positions = jnp.arange(1, k + 1)
        precision_at_k = cumsum_rel / positions

        # AP = sum(precision@k * rel(k)) / total_relevant
        # Only count positions where item is relevant
        ap_sum = (precision_at_k * top_k_binary).sum(axis=-1)
        total_relevant = (labels > 0).sum(axis=-1)

        # Handle queries with no relevant items
        ap = jnp.where(total_relevant > 0, ap_sum / total_relevant, 0.0)

        self.total_ap[...] += ap.sum()
        self.num_queries[...] += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        """Compute and return the mean average precision."""
        return self.total_ap[...] / self.num_queries[...]


class NDCG(nnx.Metric):
    """Normalized Discounted Cumulative Gain.

    Args:
        k: Number of top items to consider. If None, considers all items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import NDCG
        >>>
        >>> scores    = jnp.array([0.1, 0.4, 0.3, 0.2])
        >>> relevance = jnp.array([  3,   2,   1,   0])
        >>> metric = NDCG(k=3)
        >>> metric.update(labels=relevance, scores=scores)
        >>> metric.compute()  # DCG / IDCG
        Array(0.5525..., dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, labels: jnp.ndarray, scores: jnp.ndarray, **_) -> None:  # pyright: ignore[reportIncompatibleMethodOverride]
        """Update the NDCG with a batch of scored items.

        Args:
            labels: Relevance labels (can be graded), shape :code:`(..., num_items)`.
            scores: Scores for each item, same shape as labels.
        """
        # Flatten all batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        labels = labels.reshape(-1, original_shape[-1])

        # Cap k at the number of items available
        k = min(self.k, scores.shape[-1]) if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Compute DCG: sum of relevance / log2(rank + 1)
        ranks = jnp.arange(1, k + 1)
        discounts = jnp.log2(ranks + 1)
        dcg = (top_k_relevance / discounts).sum(axis=-1)

        # Compute IDCG: DCG for ideal ranking (sorted by relevance)
        _, ideal_indices = lax.top_k(labels, k)
        ideal_relevance = jnp.take_along_axis(labels, ideal_indices, axis=-1)
        idcg = (ideal_relevance / discounts).sum(axis=-1)

        # NDCG = DCG / IDCG (handle zero IDCG)
        ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)

        self.total_ndcg[...] += ndcg.sum()
        self.count[...] += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        """Compute and return the NDCG."""
        return self.total_ndcg[...] / self.count[...]
