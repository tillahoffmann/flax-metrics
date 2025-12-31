import inspect

from flax import nnx

import flax_metrics  # noqa: F401


def get_all_subclasses(cls):
    subs = set(cls.__subclasses__())
    for sub in list(subs):
        subs.update(get_all_subclasses(sub))
    return subs


# Override __signature__ for nnx.Metric subclasses to fix autodoc picking up
# the metaclass's generic (*args, **kwargs) signature instead of __init__.
for cls in get_all_subclasses(nnx.Metric):
    sig = inspect.signature(cls.__init__)
    # Remove 'self' parameter
    params = [p for p in sig.parameters.values() if p.name != "self"]
    cls.__signature__ = sig.replace(parameters=params)

project = "Flax Metrics"
html_theme = "sphinx_book_theme"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
]
