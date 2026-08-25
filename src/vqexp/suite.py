from __future__ import annotations

import argparse
from pathlib import Path

from .config import experiment_names, load_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect experiment suites.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List experiments.")
    list_parser.add_argument("--suite", required=True, type=Path)
    list_parser.add_argument("--group", default=None)

    args = parser.parse_args()

    if args.command == "list":
        suite = load_suite(args.suite)
        names = experiment_names(suite, group=args.group)
        for name in names:
            print(name)


if __name__ == "__main__":
    main()
