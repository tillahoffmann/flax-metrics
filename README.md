# 📏 Flax Metrics [![Flax Metrics](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/flax-metrics.svg)](https://pypi.org/project/flax-metrics/) [![Documentation Status](https://readthedocs.org/projects/flax-metrics/badge/?version=latest)](https://flax-metrics.readthedocs.io/)

Flax NXX implementation of common metrics. See the [documentation](https://flax-metrics.readthedocs.io/) for a comprehensive list of available metrics.

```python
>>> from flax_metrics import Precision, Recall
>>> from jax import numpy as jnp

>>> labels = jnp.asarray([ 0,  0,  0,  1,  1,  1])
>>> logits = jnp.asarray([-1, -2,  2,  1, -1, -2])

>>> metric = Recall()
>>> metric.update(labels=labels, logits=logits)
Recall(...)
>>> metric.compute()
Array(0.333..., dtype=float32)

```

## Masking

`jax.jit` requires re-compilation for arrays of different shapes, making evaluation on subsets challenging—we cannot index arrays with a mask. Flax Metrics supports masking through the keyword-only argument `mask`. The example below illustrates that passing `mask` is equivalent to indexing the input with a binary mask.

```python
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

```

## Chaining

Metric creation, updates, and computation can be combined into one expression by chaining operations.

```python
>>> Recall().update(labels=labels, logits=logits).compute()
Array(0.333..., dtype=float32)

```
