"""Reject a vacuous scientific pass: missing or misspelled checks cannot certify a model."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from rulfuse.sanity import run_checks
from tests.factories import MICRO

pytestmark = [pytest.mark.unit, pytest.mark.adversarial, pytest.mark.requirements("P14", "P20")]


@pytest.mark.parametrize("checks", [[], [8]])
def test_sanity_gate_refuses_to_pass_without_any_valid_checks(tmp_path, checks):
    with pytest.raises(ValueError):
        run_checks(MagicMock(), None, None, MICRO, tmp_path, checks)
