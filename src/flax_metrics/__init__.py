from flax.nnx import metrics

from .binary import Precision, Recall

Accuracy = metrics.Accuracy
Average = metrics.Average
Welford = metrics.Welford


__all__ = [
    "Accuracy",
    "Average",
    "Precision",
    "Recall",
    "Welford",
]
