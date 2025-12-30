"""Metrics for evaluating ranked retrieval where scores are computed as dot products
between query and key embeddings. These are useful for dense retrieval and
embedding-based recommendation systems where computing all pairwise scores is
prohibitive, so only a sampled subset of candidates is evaluated.
"""

from flax import nnx
from jax import lax
from jax import numpy as jnp


def _compute_dot_product_scores(
    query: jnp.ndarray, keys: jnp.ndarray, indices: jnp.ndarray
) -> jnp.ndarray:
    """Compute dot product scores for indexed keys.

    Args:
        query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
        keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
        indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.

    Returns:
        Scores with shape :code:`(*batch_shape, num_sampled)`.
    """
    subset_keys = keys[indices]  # (*batch_shape, num_sampled, num_features)
    return jnp.einsum("...f,...nf->...n", query, subset_keys)


class DotProductPrecisionAtK(nnx.Metric):
    """Precision@K using dot product scores between query and key embeddings.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

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
        """Reset the metric state in-place."""
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
        """Update the precision@k with a batch of query/key embeddings.

        Args:
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            relevance: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
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
        """Compute and return the precision@k."""
        return self.relevant_in_top_k.value / self.total_items_considered.value


class DotProductRecallAtK(nnx.Metric):
    """Recall@K using dot product scores between query and key embeddings.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider.
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))
        self.total_relevant = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
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
        """Update the recall@k with a batch of query/key embeddings.

        Args:
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            relevance: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        num_sampled = scores.shape[-1]
        effective_k = min(self.k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(relevance, top_k_indices, axis=-1)

        self.relevant_in_top_k.value += top_k_relevance.sum()
        self.total_relevant.value += relevance.sum()

    def compute(self) -> jnp.ndarray:
        """Compute and return the recall@k."""
        return self.relevant_in_top_k.value / self.total_relevant.value


class DotProductMeanReciprocalRank(nnx.Metric):
    """Mean Reciprocal Rank using dot product scores between query and key embeddings.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
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
        """Update the mean reciprocal rank with a batch of query/key embeddings.

        Args:
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            relevance: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
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
        """Compute and return the mean reciprocal rank."""
        return self.total_rr.value / self.num_queries.value


class DotProductMeanAveragePrecision(nnx.Metric):
    """Mean Average Precision using dot product scores between query and key embeddings.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
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
        """Update the mean average precision with a batch of query/key embeddings.

        Args:
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            relevance: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
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
        """Compute and return the mean average precision."""
        return self.total_ap.value / self.num_queries.value


class DotProductNDCG(nnx.Metric):
    """Normalized Discounted Cumulative Gain using dot product scores between query and key embeddings.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.int32))

    def reset(self) -> None:
        """Reset the metric state in-place."""
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
        """Update the NDCG with a batch of query/key embeddings.

        Args:
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            relevance: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
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
        """Compute and return the NDCG."""
        return self.total_ndcg.value / self.count.value
