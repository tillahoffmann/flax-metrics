"""Metrics for evaluating ranked retrieval where scores are computed as dot products
between query and key embeddings. These are useful for dense retrieval and
embedding-based recommendation systems where computing all pairwise scores is
prohibitive, so only a sampled subset of candidates is evaluated.
"""

from typing import Self

from flax import nnx
from jax import lax
from jax import numpy as jnp

from .base import BaseMetric


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


class DotProductPrecisionAtK(BaseMetric):
    """Precision@K using dot product scores between query and key embeddings.

    .. seealso::
        This metric is implemented in ir-measures as :ref:`P <ir_measures:measures.p>`.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import DotProductPrecisionAtK
        >>>
        >>> query = jnp.array([1.0, 0.0])
        >>> keys = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        >>> indices = jnp.array([0, 1, 2])
        >>> relevance = jnp.array([1, 0, 1])
        >>> metric = DotProductPrecisionAtK(k=2)
        >>> metric.update(labels=relevance, query=query, keys=keys, indices=indices)
        DotProductPrecisionAtK(...)
        >>> metric.compute()  # top-2 by score are indices 0 (relevant), 1 (not)
        Array(0.5, dtype=float32)
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.relevant_in_top_k = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )
        self.total_items_considered = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.relevant_in_top_k = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )
        self.total_items_considered = nnx.metrics.MetricState(
            jnp.array(0, dtype=jnp.float32)
        )
        return self

    def update(
        self,
        labels: jnp.ndarray,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the precision@k with a batch of query/key embeddings.

        Args:
            labels: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            mask: Binary mask indicating which queries to include.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        if mask is None:
            mask = jnp.ones(scores.shape[:-1])

        num_sampled = scores.shape[-1]
        effective_k = min(self.k, num_sampled)

        # Get top-k indices along last axis
        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Binary relevance: any value > 0 is relevant
        # Apply mask by broadcasting to (..., k)
        self.relevant_in_top_k[...] += ((top_k_relevance > 0) * mask[..., None]).sum()
        self.total_items_considered[...] += mask.sum() * self.k
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the precision@k."""
        return self.relevant_in_top_k[...] / self.total_items_considered[...]


class DotProductRecallAtK(BaseMetric):
    """Recall@K using dot product scores between query and key embeddings.

    Computes mean recall over all queries (macro-average).

    .. seealso::
        This metric is implemented in ir-measures as :ref:`R <ir_measures:measures.r>`.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import DotProductRecallAtK
        >>>
        >>> query = jnp.array([1.0, 0.0])
        >>> keys = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        >>> indices = jnp.array([0, 1, 2])
        >>> relevance = jnp.array([1, 1, 1])
        >>> metric = DotProductRecallAtK(k=2)
        >>> metric.update(labels=relevance, query=query, keys=keys, indices=indices)
        DotProductRecallAtK(...)
        >>> metric.compute()  # 2 of 3 relevant items in top-2
        Array(0.6666667, dtype=float32)
    """

    def __init__(self, k: int) -> None:
        self.k = k
        self.total_recall = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.total_recall = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        return self

    def update(
        self,
        labels: jnp.ndarray,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the recall@k with a batch of query/key embeddings.

        Args:
            labels: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            mask: Binary mask indicating which queries to include.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        if mask is None:
            mask = jnp.ones(scores.shape[:-1])

        num_sampled = scores.shape[-1]
        effective_k = min(self.k, num_sampled)

        # Flatten batch dimensions to (num_queries, num_sampled)
        scores = scores.reshape(-1, num_sampled)
        labels = labels.reshape(-1, num_sampled)
        mask = mask.reshape(-1)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        # Compute per-query recall (binary relevance: any value > 0 is relevant)
        relevant_in_top_k = (top_k_relevance > 0).sum(axis=-1)
        total_relevant = (labels > 0).sum(axis=-1)

        # Handle queries with no relevant items (avoid division by zero)
        recall_per_query = jnp.where(
            total_relevant > 0, relevant_in_top_k / total_relevant, 0.0
        )

        self.total_recall[...] += (recall_per_query * mask).sum()
        self.num_queries[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the recall@k."""
        return self.total_recall[...] / self.num_queries[...]


class DotProductMeanReciprocalRank(BaseMetric):
    """Mean Reciprocal Rank using dot product scores between query and key embeddings.

    .. seealso::
        This metric is implemented in ir-measures as :ref:`RR <ir_measures:measures.rr>`.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import DotProductMeanReciprocalRank
        >>>
        >>> query = jnp.array([1.0, 0.0])
        >>> keys = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        >>> indices = jnp.array([0, 1, 2])
        >>> relevance = jnp.array([0, 0, 1])
        >>> metric = DotProductMeanReciprocalRank()
        >>> metric.update(labels=relevance, query=query, keys=keys, indices=indices)
        DotProductMeanReciprocalRank(...)
        >>> metric.compute()  # first relevant at rank 3
        Array(0.33333334, dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.total_rr = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        return self

    def update(
        self,
        labels: jnp.ndarray,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the mean reciprocal rank with a batch of query/key embeddings.

        Args:
            labels: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            mask: Binary mask indicating which queries to include.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        if mask is None:
            mask = jnp.ones(scores.shape[:-1])

        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        labels = labels.reshape(-1, num_sampled)
        mask = mask.reshape(-1)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        is_relevant = top_k_relevance > 0
        first_relevant_idx = jnp.argmax(is_relevant, axis=-1)
        has_relevant = jnp.any(is_relevant, axis=-1)

        reciprocal_rank = jnp.where(
            has_relevant,
            1.0 / (first_relevant_idx + 1),
            0.0,
        )

        self.total_rr[...] += (reciprocal_rank * mask).sum()
        self.num_queries[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the mean reciprocal rank."""
        return self.total_rr[...] / self.num_queries[...]


class DotProductMeanAveragePrecision(BaseMetric):
    """Mean Average Precision using dot product scores between query and key embeddings.

    .. seealso::
        This metric is implemented in ir-measures as :ref:`AP <ir_measures:measures.ap>`.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import DotProductMeanAveragePrecision
        >>>
        >>> query = jnp.array([1.0, 0.0])
        >>> keys = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        >>> indices = jnp.array([0, 1, 2])
        >>> relevance = jnp.array([1, 0, 1])
        >>> metric = DotProductMeanAveragePrecision()
        >>> metric.update(labels=relevance, query=query, keys=keys, indices=indices)
        DotProductMeanAveragePrecision(...)
        >>> metric.compute()  # (1/1 + 2/3) / 2
        Array(0.8333334, dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.total_ap = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.num_queries = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        return self

    def update(
        self,
        labels: jnp.ndarray,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the mean average precision with a batch of query/key embeddings.

        Args:
            labels: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            mask: Binary mask indicating which queries to include.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        if mask is None:
            mask = jnp.ones(scores.shape[:-1])

        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        labels = labels.reshape(-1, num_sampled)
        mask = mask.reshape(-1)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        top_k_binary = (top_k_relevance > 0).astype(jnp.float32)
        cumsum_rel = jnp.cumsum(top_k_binary, axis=-1)
        positions = jnp.arange(1, effective_k + 1)
        precision_at_k = cumsum_rel / positions

        ap_sum = (precision_at_k * top_k_binary).sum(axis=-1)
        total_relevant = (labels > 0).sum(axis=-1)

        ap = jnp.where(total_relevant > 0, ap_sum / total_relevant, 0.0)

        self.total_ap[...] += (ap * mask).sum()
        self.num_queries[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the mean average precision."""
        return self.total_ap[...] / self.num_queries[...]


class DotProductNDCG(BaseMetric):
    """Normalized Discounted Cumulative Gain using dot product scores between query and key embeddings.

    .. seealso::
        This metric is implemented in ir-measures as :ref:`nDCG <ir_measures:measures.ndcg>`.

    .. note::
        The ranked score is computed as :code:`query @ keys[indices].T`, where
        :code:`query` are embeddings with shape :code:`(..., num_features)` and
        :code:`keys` are embeddings with shape :code:`(num_candidates, num_features)`.
        When the number of candidates is large, we only consider a subset of them,
        indicated by :code:`indices` with shape :code:`(..., num_sampled)`. :code:`...`
        indicates batch dimensions that are broadcastable.

    Args:
        k: Number of top items to consider. If None, considers all sampled items.

    Example:

        >>> from jax import numpy as jnp
        >>> from flax_metrics import DotProductNDCG
        >>>
        >>> query = jnp.array([1.0, 0.0])
        >>> keys = jnp.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
        >>> indices = jnp.array([0, 1, 2])
        >>> relevance = jnp.array([1, 3, 2])
        >>> metric = DotProductNDCG()
        >>> metric.update(labels=relevance, query=query, keys=keys, indices=indices)
        DotProductNDCG(...)
        >>> metric.compute()  # DCG / IDCG
        Array(0.8174..., dtype=float32)
    """

    def __init__(self, k: int | None = None) -> None:
        self.k = k
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))

    def reset(self) -> Self:
        """Reset the metric state in-place."""
        self.total_ndcg = nnx.metrics.MetricState(jnp.array(0.0, dtype=jnp.float32))
        self.count = nnx.metrics.MetricState(jnp.array(0, dtype=jnp.float32))
        return self

    def update(
        self,
        labels: jnp.ndarray,
        query: jnp.ndarray,
        keys: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        mask: jnp.ndarray | None = None,
        **_,
    ) -> Self:
        """Update the NDCG with a batch of query/key embeddings.

        Args:
            labels: Relevance labels for indexed items, shape :code:`(*batch_shape, num_sampled)`.
            query: Query embeddings, shape :code:`(*batch_shape, num_features)`.
            keys: Key embeddings for all candidates, shape :code:`(num_candidates, num_features)`.
            indices: Indices into keys for each query, shape :code:`(*batch_shape, num_sampled)`.
            mask: Binary mask indicating which queries to include.
        """
        scores = _compute_dot_product_scores(query, keys, indices)
        if mask is None:
            mask = jnp.ones(scores.shape[:-1])

        num_sampled = scores.shape[-1]

        # Flatten batch dimensions
        scores = scores.reshape(-1, num_sampled)
        labels = labels.reshape(-1, num_sampled)
        mask = mask.reshape(-1)

        k = self.k if self.k is not None else num_sampled
        effective_k = min(k, num_sampled)

        _, top_k_indices = lax.top_k(scores, effective_k)
        top_k_relevance = jnp.take_along_axis(labels, top_k_indices, axis=-1)

        ranks = jnp.arange(1, effective_k + 1)
        discounts = jnp.log2(ranks + 1)
        dcg = (top_k_relevance / discounts).sum(axis=-1)

        _, ideal_indices = lax.top_k(labels, effective_k)
        ideal_relevance = jnp.take_along_axis(labels, ideal_indices, axis=-1)
        idcg = (ideal_relevance / discounts).sum(axis=-1)

        ndcg = jnp.where(idcg > 0, dcg / idcg, 0.0)

        self.total_ndcg[...] += (ndcg * mask).sum()
        self.count[...] += mask.sum()
        return self

    def compute(self) -> jnp.ndarray:
        """Compute and return the NDCG."""
        return self.total_ndcg[...] / self.count[...]
