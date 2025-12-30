Flax Metrics
============

Flax NNX implementation of common metrics.

Binary Classification Metrics
-----------------------------

.. autosummary::
   :nosignatures:

   ~flax_metrics.binary.Recall
   ~flax_metrics.binary.Precision
   ~flax_metrics.binary.F1Score

Ranking Metrics
---------------

.. autosummary::
   :nosignatures:

   ~flax_metrics.ranking.PrecisionAtK
   ~flax_metrics.ranking.RecallAtK
   ~flax_metrics.ranking.MeanReciprocalRank
   ~flax_metrics.ranking.MeanAveragePrecision
   ~flax_metrics.ranking.NDCG
   ~flax_metrics.ranking.DotProductPrecisionAtK
   ~flax_metrics.ranking.DotProductRecallAtK
   ~flax_metrics.ranking.DotProductMeanReciprocalRank
   ~flax_metrics.ranking.DotProductMeanAveragePrecision
   ~flax_metrics.ranking.DotProductNDCG

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   binary
   ranking
