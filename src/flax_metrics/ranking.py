from flax import nnx
from jax import lax
from jax import numpy as jnp


class RecallAtK(nnx.Metric):
    """Recall@K metric for ranking problems.

    Measures the fraction of relevant items that appear in the top-k ranked results.

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
