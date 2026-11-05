"""OmniFuser-RUL, baseline B6 (PROJECT_PLAN.md D20, docs/OMNIFUSER_RUL_PLAN.md).

The OmniFuser architecture (frozen EfficientNet-B0 photo features, MRTE, image GAT, C2F
fusion with refinement) with one RUL output, on this project's segment store, folds,
point weights, selection and metrics. A separate model: no RUL-Fuse decision reads it.

Generic like the core (P22): it reads a :class:`~rulfuse.core.store.SegmentStore` and a
signal source, never an adapter; image types are the store's ``kind`` values.
"""
