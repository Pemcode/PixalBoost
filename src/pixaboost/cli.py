"""Command line entry point."""

from __future__ import annotations

import argparse

from pixaboost import __version__
from pixaboost.trials.cli_single_view import (
    configure_single_view_cli,
    dispatch_single_view_cli,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pixaboost", description=__doc__)
    parser.add_argument("--version", action="version", version=__version__)
    configure_single_view_cli(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = dispatch_single_view_cli(args)
    if result is not None:
        return result
    parser.print_help()
    return 0
