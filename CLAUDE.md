This project implements common metrics for the Flax NNX ecosystem with native support for jit-compiling.

All metrics inherit from `flax_metrics.base.BaseMetric` and must implement the following methods:

- `__init__(self, ...)`: Initialize the metric with configuration variables `...`, e.g., the depth `k` for `RecallAtK` or `threshold` on logits for binary `Recall`.
- `update(self, <target>, <prediction>, *, mask=None)`: Update the state of the metric in-place and return the metric; argument names for target and prediction may vary by implementation.
    - The target represents the ground truth to evaluate against, and the prediction may be a point estimate, logits for classification, scores for ranking, etc.
    - `mask` is a keyword-only argument indicating which elements to include. The implementation must ensure `update(target, prediction, mask=mask)` is equivalent to `update(target[mask], prediction[mask])`. This ensures that jitted `update`s are not re-compiled for different masks.
    - Calling `update` multiple times should usually be equivalent to calling `update` once with the concatenated data. This behavior is not mandatory, e.g., exponential moving averages may exhibit different behavior for multiple calls.
- `compute(self)`: Compute and return the metric *value*.
- `reset(self)`: Reset the state of the metric in-place and return the metric.

# Guidelines

- You MUST use `uv run ...` to execute commands, including Git commands.
- You MUST run `uv run pre-commit run --all-files` before attempting a commit.
- Write basic functional tests, do not write exhaustive tests for all edge cases.
- All methods must be jit-able, i.e., they cannot use arrays with shape that depend on runtime variables.
