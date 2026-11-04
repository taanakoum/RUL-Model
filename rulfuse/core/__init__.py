"""The dataset-agnostic core (PROJECT_PLAN.md D18, §6.11).

Everything here works on the generic schema of :mod:`rulfuse.core.schema`: units with
recordings, images and a failure time. Nothing in this package knows what a pass, run,
block, tool photo or flank wear is, and nothing here imports :mod:`rulfuse.adapters`
(test P22 checks both).
"""
