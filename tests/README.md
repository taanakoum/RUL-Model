# Test suite

Run and category documentation: [docs/TESTING.md](../docs/TESTING.md).

New checks are grouped into `unit/`, `integration/`, `e2e/`, `data/`, and `gpu/`.
Existing test paths are preserved and classified by `catalog.py` and `conftest.py`.

The current protocol uses seed 0. Tests marked `compatibility` exercise retained
multi-seed APIs separately; they do not impose a three-seed production requirement.

Failures in the new strict acceptance tests are actionable results, not expected-failure
placeholders. Do not weaken them to match defective behavior. See the dated review in
`docs/TEST_REVIEW_2026-11-15.md` for the initial findings, priorities and coverage limits.
