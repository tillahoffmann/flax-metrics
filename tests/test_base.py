import jax.numpy as jnp
import numpy as np
import pytest
from conftest import update_and_compute, validate_masking
from numpy.testing import assert_almost_equal

from flax_metrics import Accuracy, Average, Statistics, Welford


@pytest.mark.parametrize(
    "values, expected",
    [
        ([1.0, 2.0, 3.0, 4.0], 2.5),
        ([0.0, 0.0, 0.0], 0.0),
        ([5.0], 5.0),
        ([-1.0, 1.0], 0.0),
        ([1.5, 2.5, 3.5], 2.5),
    ],
)
def test_average_computes_mean(values, expected, jit, masked):
    """Verify Average computes the arithmetic mean."""
    values = jnp.array(values)

    metric = Average()
    if masked:
        validate_masking(metric, (values,), {}, jit=jit, event_dim=0)
        return

    update, compute = update_and_compute(metric, jit)
    update(values)
    actual = float(compute())

    assert_almost_equal(actual, expected)


def test_average_accumulation(jit):
    """Accumulated average over batches matches numpy on combined data."""
    metric = Average()
    update, compute = update_and_compute(metric, jit)

    values1 = jnp.array([1.0, 2.0])
    update(values1)

    values2 = jnp.array([3.0, 4.0, 5.0])
    update(values2)

    actual = float(compute())

    all_values = np.concatenate([values1, values2])
    expected = np.mean(all_values)

    assert_almost_equal(actual, expected)


def test_average_reset(jit):
    """Reset clears accumulated state."""
    metric = Average()
    update, compute = update_and_compute(metric, jit)

    update(jnp.array([100.0, 200.0]))
    metric.reset()
    update(jnp.array([1.0, 2.0, 3.0]))

    actual = float(compute())
    expected = 2.0

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "logits, labels, expected",
    [
        # Multi-class: all correct
        ([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]], [1, 0, 1], 1.0),
        # Multi-class: 2/3 correct
        ([[0.9, 0.1], [0.8, 0.2], [0.3, 0.7]], [1, 0, 1], 2 / 3),
        # Multi-class: none correct (predictions: [1, 1, 1], labels: [0, 0, 0])
        ([[0.1, 0.9], [0.2, 0.8], [0.3, 0.7]], [0, 0, 0], 0.0),
    ],
)
def test_accuracy_multiclass(logits, labels, expected, jit, masked):
    """Verify multi-class accuracy computes correctly."""
    logits = jnp.array(logits)
    labels = jnp.array(labels)

    metric = Accuracy()
    if masked:
        # event_dim=1 because logits has shape (..., num_classes)
        validate_masking(
            metric, (), {"logits": logits, "labels": labels}, jit=jit, event_dim=1
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(logits=logits, labels=labels)
    actual = float(compute())

    assert_almost_equal(actual, expected, decimal=5)


@pytest.mark.parametrize(
    "logits, labels, threshold, expected",
    [
        # All correct
        ([0.6, 0.4, 0.8, 0.3], [1, 0, 1, 0], 0.5, 1.0),
        # 3/4 correct
        ([0.6, 0.6, 0.8, 0.3], [1, 0, 1, 0], 0.5, 3 / 4),
        # None correct
        ([0.4, 0.6, 0.4, 0.6], [1, 0, 1, 0], 0.5, 0.0),
    ],
)
def test_accuracy_binary(logits, labels, threshold, expected, jit, masked):
    """Verify binary accuracy computes correctly."""
    logits = jnp.array(logits)
    labels = jnp.array(labels)

    metric = Accuracy(threshold=threshold)
    if masked:
        validate_masking(
            metric, (), {"logits": logits, "labels": labels}, jit=jit, event_dim=0
        )
        return

    update, compute = update_and_compute(metric, jit)
    update(logits=logits, labels=labels)
    actual = float(compute())

    assert_almost_equal(actual, expected, decimal=5)


def test_accuracy_accumulation(jit):
    """Accumulated accuracy over batches matches numpy on combined data."""
    metric = Accuracy()
    update, compute = update_and_compute(metric, jit)

    logits1 = jnp.array([[0.1, 0.9], [0.8, 0.2]])
    labels1 = jnp.array([1, 0])
    update(logits=logits1, labels=labels1)

    logits2 = jnp.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
    labels2 = jnp.array([1, 1, 0])
    update(logits=logits2, labels=labels2)

    actual = float(compute())

    # Combined: [1, 0, 1, 1, 0] predictions vs [1, 0, 1, 1, 0] labels
    all_logits = np.concatenate([logits1, logits2])
    all_labels = np.concatenate([labels1, labels2])
    expected = np.mean(all_logits.argmax(axis=-1) == all_labels)

    assert_almost_equal(actual, expected)


def test_accuracy_reset(jit):
    """Reset clears accumulated state."""
    metric = Accuracy()
    update, compute = update_and_compute(metric, jit)

    update(logits=jnp.array([[0.1, 0.9]]), labels=jnp.array([0]))  # Wrong
    metric.reset()
    update(
        logits=jnp.array([[0.1, 0.9], [0.8, 0.2]]), labels=jnp.array([1, 0])
    )  # All correct

    actual = float(compute())
    expected = 1.0

    assert_almost_equal(actual, expected)


@pytest.mark.parametrize(
    "values",
    [
        [1.0, 2.0, 3.0, 4.0],
        [0.0, 0.0, 0.0],
        [5.0],
        [-1.0, 1.0, -2.0, 2.0],
        [1.5, 2.5, 3.5, 4.5, 5.5],
    ],
)
def test_welford_computes_statistics(values, jit, masked):
    """Verify Welford computes mean and std correctly."""
    values = jnp.array(values)

    metric = Welford()
    if masked:
        validate_masking(metric, (values,), {}, jit=jit, event_dim=0)
        return

    update, compute = update_and_compute(metric, jit)
    update(values)
    stats = compute()

    assert isinstance(stats, Statistics)
    assert_almost_equal(float(stats.mean), float(np.mean(values)), decimal=5)
    # Population std (ddof=0)
    assert_almost_equal(
        float(stats.standard_deviation), float(np.std(values)), decimal=5
    )


def test_welford_accumulation(jit):
    """Accumulated Welford over batches matches numpy on combined data."""
    metric = Welford()
    update, compute = update_and_compute(metric, jit)

    values1 = jnp.array([1.0, 2.0])
    update(values1)

    values2 = jnp.array([3.0, 4.0, 5.0])
    update(values2)

    stats = compute()

    all_values = np.concatenate([values1, values2])
    expected_mean = np.mean(all_values)
    expected_std = np.std(all_values)  # Population std

    assert_almost_equal(float(stats.mean), expected_mean, decimal=5)
    assert_almost_equal(float(stats.standard_deviation), expected_std, decimal=5)


def test_welford_reset(jit):
    """Reset clears accumulated state."""
    metric = Welford()
    update, compute = update_and_compute(metric, jit)

    update(jnp.array([100.0, 200.0]))
    metric.reset()
    update(jnp.array([1.0, 2.0, 3.0]))

    stats = compute()
    expected_mean = 2.0
    expected_std = np.std([1.0, 2.0, 3.0])

    assert_almost_equal(float(stats.mean), expected_mean, decimal=5)
    assert_almost_equal(float(stats.standard_deviation), expected_std, decimal=5)


def test_welford_sem(jit):
    """Verify Welford computes standard error of mean correctly."""
    metric = Welford()
    update, compute = update_and_compute(metric, jit)

    values = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    update(values)

    stats = compute()

    # SEM = std / sqrt(n)
    expected_sem = np.std(values) / np.sqrt(len(values))
    assert_almost_equal(float(stats.standard_error_of_mean), expected_sem, decimal=5)
