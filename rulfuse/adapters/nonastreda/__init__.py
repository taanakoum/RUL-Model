"""The Nonastreda adapter (PROJECT_PLAN.md §6.11): everything specific to this dataset.

``load()`` returns the generic :class:`rulfuse.core.schema.Dataset`; ``load_data()`` the
adapter's own view (passes, runs, labels) for audits and the EDA.
"""

from rulfuse.adapters.nonastreda.dataset import (
    NonastredaData,
    NonastredaOptions,
    load,
    load_data,
    to_dataset,
)

__all__ = ["NonastredaData", "NonastredaOptions", "load", "load_data", "to_dataset"]
