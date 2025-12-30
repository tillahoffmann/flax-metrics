from flax.nnx import metrics

from .binary import F1Score, Precision, Recall
from .ranking import (
    NDCG,
    DotProductMeanAveragePrecision,
    DotProductMeanReciprocalRank,
    DotProductNDCG,
    DotProductPrecisionAtK,
    DotProductRecallAtK,
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
    "MeanAveragePrecision",
    "MeanReciprocalRank",
    "NDCG",
    "Precision",
    "PrecisionAtK",
    "Recall",
    "RecallAtK",
    "Welford",
]
