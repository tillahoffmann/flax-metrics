Flax Metrics
============

Flax NNX implementation of common metrics.

Binary Classification Metrics
-----------------------------

Metrics for evaluating binary classifiers, operating on logits and binary
labels.

.. autosummary::
   :nosignatures:

   ~flax_metrics.binary.Recall
   ~flax_metrics.binary.Precision
   ~flax_metrics.binary.F1Score

Ranking Metrics
---------------

Metrics for evaluating ranked retrieval results using precomputed scores.

.. autosummary::
   :nosignatures:

   ~flax_metrics.ranking.PrecisionAtK
   ~flax_metrics.ranking.RecallAtK
   ~flax_metrics.ranking.MeanReciprocalRank
   ~flax_metrics.ranking.MeanAveragePrecision
   ~flax_metrics.ranking.NDCG

Dot Product Ranking Metrics
---------------------------

Ranking metrics where scores are computed as dot products between query and key
embeddings. Useful for dense retrieval and embedding-based recommendation
systems.

.. autosummary::
   :nosignatures:

   ~flax_metrics.dot_product_ranking.DotProductPrecisionAtK
   ~flax_metrics.dot_product_ranking.DotProductRecallAtK
   ~flax_metrics.dot_product_ranking.DotProductMeanReciprocalRank
   ~flax_metrics.dot_product_ranking.DotProductMeanAveragePrecision
   ~flax_metrics.dot_product_ranking.DotProductNDCG

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   binary
   ranking
   dot_product_ranking
