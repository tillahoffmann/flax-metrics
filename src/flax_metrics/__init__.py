from flax.nnx import metrics

from .classification import F1Score, LogProb, Precision, Recall
from .dot_product_ranking import (
    DotProductMeanAveragePrecision,
    DotProductMeanReciprocalRank,
    DotProductNDCG,
    DotProductPrecisionAtK,
    DotProductRecallAtK,
)
from .ranking import (
    NDCG,
    MeanAveragePrecision,
    MeanReciprocalRank,
    PrecisionAtK,
    RecallAtK,
)

Accuracy = metrics.Accuracy
Average = metrics.Average
Welford = metrics.Welford


__all__ = [
    "Accuracy",
    "Average",
    "DotProductMeanAveragePrecision",
    "DotProductMeanReciprocalRank",
    "DotProductNDCG",
    "DotProductPrecisionAtK",
    "DotProductRecallAtK",
    "F1Score",
    "LogProb",
    "MeanAveragePrecision",
    "MeanReciprocalRank",
    "NDCG",
    "Precision",
    "PrecisionAtK",
    "Recall",
    "RecallAtK",
    "Welford",
]
