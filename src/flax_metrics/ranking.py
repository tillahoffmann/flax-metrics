from flax import nnx
from jax import lax
from jax import numpy as jnp


class PrecisionAtK(nnx.Metric):
    """Precision@K, the fraction of top-k items that are relevant.

    Args:
        k: Number of top items to consider.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, *, scores: jnp.ndarray, relevance: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of queries.

        Args:
            scores: Scores for each item, shape (..., num_items).
            relevance: Binary relevance labels, same shape as scores.
        """
        # Flatten batch dimensions to count queries
        num_queries = scores.size // scores.shape[-1]

        # Get top-k indices along last axis (descending order)
        _, top_k_indices = lax.top_k(scores, self.k)

        # Gather relevance values for top-k items
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        # Accumulate counts
        self.relevant_in_top_k.value += top_k_relevance.sum()
        self.num_queries.value += num_queries

    def compute(self) -> jnp.ndarray:
        return self.relevant_in_top_k.value / (self.num_queries.value * self.k)


class RecallAtK(nnx.Metric):
    """Recall@K, the fraction of relevant items that appear in the top-k ranked results.

    Args:
        k: Number of top items to consider.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.total_relevant = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.total_relevant = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, *, scores: jnp.ndarray, relevance: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of queries.

        Args:
            scores: Scores for each item, shape (..., num_items).
            relevance: Binary relevance labels, same shape as scores.
        """
        # Get top-k indices along last axis (descending order)
        _, top_k_indices = lax.top_k(scores, self.k)

        # Gather relevance values for top-k items
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        # Accumulate counts
        self.relevant_in_top_k.value += top_k_relevance.sum()
        self.total_relevant.value += relevance.sum()

    def compute(self) -> jnp.ndarray:
        return self.relevant_in_top_k.value / self.total_relevant.value


class MRR(nnx.Metric):
    """Mean Reciprocal Rank.

    The average of reciprocal ranks of the first relevant item for each query.

    Args:
        k: Number of top items to consider. If None, considers all items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, *, scores: jnp.ndarray, relevance: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of queries.

        Args:
            scores: Scores for each item, shape (..., num_items).
            relevance: Binary relevance labels, same shape as scores.
        """
        # Flatten batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        relevance = relevance.reshape(-1, original_shape[-1])

        k = self.k if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        # Find rank of first relevant item (1-indexed)
        # Use argmax on relevance; if no relevant item, argmax returns 0
        first_relevant_idx = jnp.argmax(top_k_relevance, axis=-1)
        has_relevant = top_k_relevance.sum(axis=-1) > 0

        # Reciprocal rank: 1/(rank), where rank = index + 1
        reciprocal_rank = jnp.where(
            has_relevant
            & (top_k_relevance[jnp.arange(scores.shape[0]), first_relevant_idx] > 0),
            1.0 / (first_relevant_idx + 1),
            0.0,
        )

        self.total_rr.value += reciprocal_rank.sum()
        self.num_queries.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_rr.value / self.num_queries.value


class NDCG(nnx.Metric):
    """Normalized Discounted Cumulative Gain.

    Args:
        k: Number of top items to consider. If None, uses all items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, *, scores: jnp.ndarray, relevance: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of queries.

        Args:
            scores: Scores for each item, shape (..., num_items).
            relevance: Relevance labels (can be graded), same shape as scores.
        """
        # Flatten all batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        relevance = relevance.reshape(-1, original_shape[-1])

        k = self.k if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        # Compute DCG: sum of relevance / log2(rank + 1)
        ranks = jnp.arange(1, k + 1)
        discounts = jnp.log2(ranks + 1)
        dcg = (top_k_relevance / discounts).sum(axis=-1)

        # Compute IDCG: DCG for ideal ranking (sorted by relevance)
        _, ideal_indices = lax.top_k(relevance, k)
        ideal_relevance = jnp.take_along_axis(relevance, ideal_indices, axis=-1)
        idcg = (ideal_relevance / discounts).sum(axis=-1)

        # NDCG = DCG / IDCG (handle zero IDCG)
        ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)

        self.total_ndcg.value += ndcg.sum()
        self.count.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_ndcg.value / self.count.value
