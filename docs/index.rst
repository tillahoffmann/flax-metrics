📏 Flax Metrics
===============

Flax NNX implementation of common metrics.

.. doctest::

   >>> from flax_metrics import Precision, Recall
   >>> from jax import numpy as jnp

   >>> labels = jnp.asarray([ 0,  0,  0,  1,  1,  1])
   >>> logits = jnp.asarray([-1, -2,  2,  1, -1, -2])

   >>> metric = Recall()
   >>> metric.update(labels=labels, logits=logits)
   Recall(...)
   >>> metric.compute()
   Array(0.333..., dtype=float32)

Masking
-------

:func:`jax.jit` requires re-compilation for arrays of different shapes, making evaluation on subsets challenging—we cannot index arrays with a mask. Flax Metrics supports masking through the keyword-only argument :code:`mask`. The example below illustrates that passing :code:`mask` is equivalent to indexing the input with a binary mask.

.. doctest::

   >>> mask = jnp.asarray([True, True, True, True, False, True])
   >>> metric = Recall()
   >>> metric.update(labels=labels, logits=logits, mask=mask)
   Recall(...)
   >>> metric.compute()
   Array(0.5, dtype=float32)

   >>> metric.reset()
   Recall(...)
   >>> metric.update(labels=labels[mask], logits=logits[mask])
   Recall(...)
   >>> metric.compute()
   Array(0.5, dtype=float32)

Available Metrics
-----------------

Classification Metrics
^^^^^^^^^^^^^^^^^^^^^^

Metrics for evaluating classifiers, operating on logits and binary, categorical, or multinomial labels.

.. autosummary::
   :nosignatures:

   ~flax_metrics.classification.Recall
   ~flax_metrics.classification.Precision
   ~flax_metrics.classification.F1Score
   ~flax_metrics.classification.LogProb

Ranking Metrics
^^^^^^^^^^^^^^^

Metrics for evaluating ranked retrieval results using precomputed scores.

.. autosummary::
   :nosignatures:

   ~flax_metrics.ranking.PrecisionAtK
   ~flax_metrics.ranking.RecallAtK
   ~flax_metrics.ranking.MeanReciprocalRank
   ~flax_metrics.ranking.MeanAveragePrecision
   ~flax_metrics.ranking.NDCG

Dot Product Ranking Metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Ranking metrics where scores are computed as dot products between query and key embeddings. Useful for dense retrieval and embedding-based recommendation systems.

.. autosummary::
   :nosignatures:

   ~flax_metrics.dot_product_ranking.DotProductPrecisionAtK
   ~flax_metrics.dot_product_ranking.DotProductRecallAtK
   ~flax_metrics.dot_product_ranking.DotProductMeanReciprocalRank
   ~flax_metrics.dot_product_ranking.DotProductMeanAveragePrecision
   ~flax_metrics.dot_product_ranking.DotProductNDCG

Base Metrics
^^^^^^^^^^^^

General-purpose metrics for aggregating values.

.. autosummary::
   :nosignatures:

   ~flax_metrics.base.Accuracy
   ~flax_metrics.base.Average
   ~flax_metrics.base.Statistics
   ~flax_metrics.base.Welford

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   base
   classification
   ranking
   dot_product_ranking
