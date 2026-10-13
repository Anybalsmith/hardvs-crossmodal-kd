"""mmaction2 plugins for RGB -> Event cross-modal distillation on HARDVS.

Importing this package registers its datasets, transforms, models and hooks
in the mmaction registries; configs do it through ``custom_imports``.
"""
from . import datasets, engine, models  # noqa: F401
