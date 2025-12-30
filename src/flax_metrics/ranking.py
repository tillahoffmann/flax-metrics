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
            relevance: Relevance labels, same shape as scores.
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
            relevance: Relevance labels, same shape as scores.
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


class MeanReciprocalRank(nnx.Metric):
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
            relevance: Relevance labels, same shape as scores.
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
        is_relevant = top_k_relevance > 0
        first_relevant_idx = jnp.argmax(is_relevant, axis=-1)
        has_relevant = jnp.any(is_relevant, axis=-1)

        # Reciprocal rank: 1/(rank), where rank = index + 1
        reciprocal_rank = jnp.where(
            has_relevant,
            1.0 / (first_relevant_idx + 1),
            0.0,
        )

        self.total_rr.value += reciprocal_rank.sum()
        self.num_queries.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_rr.value / self.num_queries.value


class MeanAveragePrecision(nnx.Metric):
    """Mean Average Precision.

    The mean of average precision scores across queries, where average precision
    is the sum of precision@k * rel(k) divided by total relevant items.

    Args:
        k: Number of top items to consider. If None, considers all items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(self, *, scores: jnp.ndarray, relevance: jnp.ndarray, **_) -> None:
        """Update the metric with a batch of queries.

        Args:
            scores: Scores for each item, shape (..., num_items).
            relevance: Relevance labels, same shape as scores.
        """
        # Flatten batch dimensions
        original_shape = scores.shape
        scores = scores.reshape(-1, original_shape[-1])
        relevance = relevance.reshape(-1, original_shape[-1])

        k = self.k if self.k is not None else scores.shape[-1]

        # Get top-k indices by score
        _, top_k_indices = lax.top_k(scores, k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

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
        total_relevant = (relevance > 0).sum(axis=-1)

        # Handle queries with no relevant items
        ap = jnp.where(total_relevant > 0, ap_sum / total_relevant, 0.0)

        self.total_ap.value += ap.sum()
        self.num_queries.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_ap.value / self.num_queries.value


class NDCG(nnx.Metric):
    """Normalized Discounted Cumulative Gain.

    Args:
        k: Number of top items to consider. If None, considers all items.
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


def _compute_dot_product_scores(
    query: jnp.ndarray, keys: jnp.ndarray, indices: jnp.ndarray
) -> jnp.ndarray:
    """Compute dot product scores for indexed keys.

    Args:
        query: Query embeddings, shape (*batch_shape, num_features).
        keys: Key embeddings for all candidates, shape (num_candidates, num_features).
        indices: Indices into keys for each query, shape (*batch_shape, num_sampled).

    Returns:
        Scores with shape (*batch_shape, num_sampled).
    """
    subset_keys = keys[indices]  # (*batch_shape, num_sampled, num_features)
    return jnp.einsum("...f,...nf->...n", query, subset_keys)


class DotProductPrecisionAtK(nnx.Metric):
    """Precision@K computed from query/key embeddings via dot product.

    The ranked score is computed as :code:`query @ keys[indices].T`, where :code:`query`
    are embeddings with shape :code:`(..., num_features)` and :code:`keys` are
    embeddings with shape :code:`(num_candidates, num_features)`. When the number of
    candidates is large, we only consider a subset of them, indicated by :code:`indices`
    with shape :code:`(..., num_sampled)`. :code:`...` indicates batch dimensions that
    are broadcastable.

    Args:
        k: Number of top items to consider.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.total_items_considered = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def reset(self) -> None:
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.total_items_considered = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.int32)
        )

    def update(
        self,
        *,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        relevance: jnp.ndarray,
        **_,
    ) -> None:
        """Update the metric with a batch of queries.

        Args:
            query: Query embeddings, shape (*batch_shape, num_features).
            keys: Key embeddings for all candidates, shape (num_candidates, num_features).
            indices: Indices into keys for each query, shape (*batch_shape, num_sampled).
            relevance: Relevance labels for indexed items, shape (*batch_shape, num_sampled).
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]
        effective_k = min(self.k, num_sampled)

        # Flatten batch dimensions
        num_queries = scores.size // num_sampled

        # Get top-k indices along last axis
        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        self.relevant_in_top_k.value += top_k_relevance.sum()
        self.total_items_considered.value += num_queries * effective_k

    def compute(self) -> jnp.ndarray:
        return self.relevant_in_top_k.value / self.total_items_considered.value


class DotProductRecallAtK(nnx.Metric):
    """Recall@K computed from query/key embeddings via dot product.

    The ranked score is computed as :code:`query @ keys[indices].T`, where :code:`query`
    are embeddings with shape :code:`(..., num_features)` and :code:`keys` are
    embeddings with shape :code:`(num_candidates, num_features)`. When the number of
    candidates is large, we only consider a subset of them, indicated by :code:`indices`
    with shape :code:`(..., num_sampled)`. :code:`...` indicates batch dimensions that
    are broadcastable.

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

    def update(
        self,
        *,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        relevance: jnp.ndarray,
        **_,
    ) -> None:
        """Update the metric with a batch of queries.

        Args:
            query: Query embeddings, shape (*batch_shape, num_features).
            keys: Key embeddings for all candidates, shape (num_candidates, num_features).
            indices: Indices into keys for each query, shape (*batch_shape, num_sampled).
            relevance: Relevance labels for indexed items, shape (*batch_shape, num_sampled).
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]
        effective_k = min(self.k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        self.relevant_in_top_k.value += top_k_relevance.sum()
        self.total_relevant.value += relevance.sum()

    def compute(self) -> jnp.ndarray:
        return self.relevant_in_top_k.value / self.total_relevant.value


class DotProductMeanReciprocalRank(nnx.Metric):
    """Mean Reciprocal Rank computed from query/key embeddings via dot product.

    The ranked score is computed as :code:`query @ keys[indices].T`, where :code:`query`
    are embeddings with shape :code:`(..., num_features)` and :code:`keys` are
    embeddings with shape :code:`(num_candidates, num_features)`. When the number of
    candidates is large, we only consider a subset of them, indicated by :code:`indices`
    with shape :code:`(..., num_sampled)`. :code:`...` indicates batch dimensions that
    are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(
        self,
        *,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        relevance: jnp.ndarray,
        **_,
    ) -> None:
        """Update the metric with a batch of queries.

        Args:
            query: Query embeddings, shape (*batch_shape, num_features).
            keys: Key embeddings for all candidates, shape (num_candidates, num_features).
            indices: Indices into keys for each query, shape (*batch_shape, num_sampled).
            relevance: Relevance labels for indexed items, shape (*batch_shape, num_sampled).
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        relevance = relevance.reshape(-1, num_sampled)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        is_relevant = top_k_relevance > 0
        first_relevant_idx = jnp.argmax(is_relevant, axis=-1)
        has_relevant = jnp.any(is_relevant, axis=-1)

        reciprocal_rank = jnp.where(
            has_relevant,
            1.0 / (first_relevant_idx + 1),
            0.0,
        )

        self.total_rr.value += reciprocal_rank.sum()
        self.num_queries.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_rr.value / self.num_queries.value


class DotProductMeanAveragePrecision(nnx.Metric):
    """Mean Average Precision computed from query/key embeddings via dot product.

    The ranked score is computed as :code:`query @ keys[indices].T`, where :code:`query`
    are embeddings with shape :code:`(..., num_features)` and :code:`keys` are
    embeddings with shape :code:`(num_candidates, num_features)`. When the number of
    candidates is large, we only consider a subset of them, indicated by :code:`indices`
    with shape :code:`(..., num_sampled)`. :code:`...` indicates batch dimensions that
    are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(
        self,
        *,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        relevance: jnp.ndarray,
        **_,
    ) -> None:
        """Update the metric with a batch of queries.

        Args:
            query: Query embeddings, shape (*batch_shape, num_features).
            keys: Key embeddings for all candidates, shape (num_candidates, num_features).
            indices: Indices into keys for each query, shape (*batch_shape, num_sampled).
            relevance: Relevance labels for indexed items, shape (*batch_shape, num_sampled).
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        relevance = relevance.reshape(-1, num_sampled)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        top_k_binary = (top_k_relevance > 0).astype(jnp.float32)
        cumsum_rel = jnp.cumsum(top_k_binary, axis=-1)
        positions = jnp.arange(1, effective_k + 1)
        precision_at_k = cumsum_rel / positions

        ap_sum = (precision_at_k * top_k_binary).sum(axis=-1)
        total_relevant = (relevance > 0).sum(axis=-1)

        ap = jnp.where(total_relevant > 0, ap_sum / total_relevant, 0.0)

        self.total_ap.value += ap.sum()
        self.num_queries.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_ap.value / self.num_queries.value


class DotProductNDCG(nnx.Metric):
    """Normalized Discounted Cumulative Gain computed from query/key embeddings via dot product.

    The ranked score is computed as :code:`query @ keys[indices].T`, where :code:`query`
    are embeddings with shape :code:`(..., num_features)` and :code:`keys` are
    embeddings with shape :code:`(num_candidates, num_features)`. When the number of
    candidates is large, we only consider a subset of them, indicated by :code:`indices`
    with shape :code:`(..., num_sampled)`. :code:`...` indicates batch dimensions that
    are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def update(
        self,
        *,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        relevance: jnp.ndarray,
        **_,
    ) -> None:
        """Update the metric with a batch of queries.

        Args:
            query: Query embeddings, shape (*batch_shape, num_features).
            keys: Key embeddings for all candidates, shape (num_candidates, num_features).
            indices: Indices into keys for each query, shape (*batch_shape, num_sampled).
            relevance: Relevance labels for indexed items, shape (*batch_shape, num_sampled).
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        relevance = relevance.reshape(-1, num_sampled)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        ranks = jnp.arange(1, effective_k + 1)
        discounts = jnp.log2(ranks + 1)
        dcg = (top_k_relevance / discounts).sum(axis=-1)

        _, ideal_indices = lax.top_k(relevance, effective_k)
        ideal_relevance = jnp.take_along_axis(relevance, ideal_indices, axis=-1)
        idcg = (ideal_relevance / discounts).sum(axis=-1)

        ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)

        self.total_ndcg.value += ndcg.sum()
        self.count.value += scores.shape[0]

    def compute(self) -> jnp.ndarray:
        return self.total_ndcg.value / self.count.value
