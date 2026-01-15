# 📏 Flax Metrics [![Flax Metrics](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/flax-metrics.svg)](https://pypi.org/project/flax-metrics/) [![Documentation Status](https://readthedocs.org/projects/flax-metrics/badge/?version=latest)](https://flax-metrics.readthedocs.io/)

Flax NXX implementation of common metrics. See the [documentation](https://flax-metrics.readthedocs.io/) for a comprehensive list of available metrics.

```python
>>> from flax_metrics import Precision, Recall
>>> from jax import numpy as jnp

>>> labels = jnp.asarray([ 0,  0,  0,  1,  1,  1])
>>> logits = jnp.asarray([-1, -2,  2,  1, -1, -2])

>>> metric = Recall()
>>> metric.update(labels=labels, logits=logits)
>>> metric.compute()
Array(0.333..., dtype=float32)

>>> metric = Precision()
>>> metric.update(labels=labels, logits=logits)
>>> metric.compute()
Array(0.5, dtype=float32)

```

`jax.jit` requires re-compilation for arrays of different shapes, making evaluation on subsets challenging—we cannot index arrays with a mask. Flax Metrics supports masking through the keyword-only argument `mask`.

```python
>>> mask = jnp.asarray([True, True, True, True, False, True])
>>> metric = Recall()
>>> metric.update(labels=labels, logits=logits, mask=mask)
>>> metric.compute()
Array(0.5, dtype=float32)

>>> metric.reset()
>>> metric.update(labels=labels[mask], logits=logits[mask])
>>> metric.compute()
Array(0.5, dtype=float32)

```
