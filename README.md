# 📏 Flax Metrics [![Flax Metrics](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml/badge.svg)](https://github.com/tillahoffmann/flax-metrics/actions/workflows/ci.yml) [![PyPI version](https://img.shields.io/pypi/v/flax-metrics.svg)](https://pypi.org/project/flax-metrics/) [![Documentation Status](https://readthedocs.org/projects/flax-metrics/badge/?version=latest)](https://flax-metrics.readthedocs.io/)

Flax NXX implementation of common metrics.

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
