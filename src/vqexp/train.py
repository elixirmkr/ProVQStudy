from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_json, materialize_config, parse_overrides, validate_config

def main() -> None:
    parser = argparse.ArgumentParser(description="Train one VQ experiment.")
    parser.add_argument("--config", type=Path, default=None, help="Single materialized config JSON.")
    parser.add_argument("--suite", type=Path, default=None, help="Suite JSON.")
    parser.add_argument("--experiment", default=None, help="Experiment name inside suite.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="Override key=value.")
    parser.add_argument("--print-config", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    overrides = parse_overrides(args.overrides)
    if args.config is not None:
        cfg = load_json(args.config)
        if args.seed is not None:
            cfg["seed"] = args.seed
        cfg.update(overrides)
        validate_config(cfg)
    else:
        if args.suite is None or args.experiment is None:
            parser.error("Either --config or both --suite and --experiment are required.")
        cfg = materialize_config(args.suite, args.experiment, seed=args.seed, overrides=overrides)

    if args.print_config or args.dry_run:
        print(json.dumps(cfg, indent=2, sort_keys=True))
    if args.dry_run:
        return

    from .runner import run_training

    run_training(cfg)


if __name__ == "__main__":
    main()
