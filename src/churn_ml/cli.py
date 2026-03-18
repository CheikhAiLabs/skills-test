from __future__ import annotations

import argparse
import json
from pathlib import Path

from churn_ml.artifacts import find_latest_bundle, load_bundle
from churn_ml.config import load_config
from churn_ml.data import write_synthetic_dataset
from churn_ml.logging_utils import configure_logging
from churn_ml.pipeline import PipelineRunner, retrain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Churn ML pipeline operations")
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline configuration YAML.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-data", help="Generate a sample dataset.")
    bootstrap.add_argument("--rows", type=int, default=5000, help="Number of rows to generate.")
    bootstrap.add_argument("--seed", type=int, default=42, help="Random seed.")

    subparsers.add_parser("run", help="Execute the full training pipeline.")

    retrain_parser = subparsers.add_parser(
        "retrain",
        help="Run multiple training rounds with candidate selection and promotion.",
    )
    retrain_parser.add_argument(
        "--rounds",
        type=int,
        default=None,
        help="Number of retraining rounds to execute.",
    )
    retrain_parser.add_argument(
        "--no-promote-best",
        action="store_true",
        help="Skip the final promotion rerun of the best round.",
    )

    report = subparsers.add_parser("report", help="Show metrics from a model bundle.")
    report.add_argument(
        "--bundle-path",
        default=None,
        help="Optional explicit bundle path. Defaults to the latest artifact bundle.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging()

    if args.command == "bootstrap-data":
        output_path = write_synthetic_dataset(
            config.data.raw_data_path,
            rows=args.rows,
            seed=args.seed,
        )
        print(output_path)
        return 0

    if args.command == "run":
        run_dir = PipelineRunner(config).run()
        print(run_dir)
        return 0

    if args.command == "retrain":
        summary = retrain(
            config,
            rounds=args.rounds or config.training.retrain_rounds,
            promote_best=not args.no_promote_best,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    if args.command == "report":
        bundle_path = (
            Path(args.bundle_path).resolve()
            if args.bundle_path
            else find_latest_bundle(config.data.artifact_root)
        )
        bundle = load_bundle(bundle_path)
        print(json.dumps(bundle["metrics"], indent=2, sort_keys=True))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
