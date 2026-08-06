"""Proves the test framework and the package layout are wired correctly (F00).

Per lecture 06, initialisation is only complete when at least one test actually
runs and passes -- otherwise a green suite may just mean nothing was collected.
"""

from __future__ import annotations

import pixaboost
from pixaboost.cli import build_parser


def test_package_exposes_a_version() -> None:
    assert pixaboost.__version__


def test_cli_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "pixaboost"
