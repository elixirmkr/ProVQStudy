from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any


METRICS = [
    "psnr",
    "ssim",
    "perplexity",
    "usage",
    "effective_rank",
    "normalized_effective_rank",
    "avg_pairwise_distance",
]


def read_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def collect_summaries(root: Path, suite_name: str) -> list[dict[str, Any]]:
    suite_root = root / suite_name
    return [read_summary(path) for path in sorted(suite_root.glob("*/*/summary.json"))]


def flatten_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        "suite_name": summary["suite_name"],
        "experiment_name": summary["experiment_name"],
        "experiment_label": summary["experiment_label"],
        "group": summary.get("group", ""),
        "seed": summary["seed"],
        "best_psnr": summary["best_psnr"],
        "best_epoch": summary["best_epoch"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "run_dir": summary["run_dir"],
    }
    final = summary.get("final_metrics") or {}
    for metric in METRICS:
        row[f"final_{metric}"] = final.get(metric)
    return row


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [value for value in values if value is not None]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], 0.0
    return statistics.mean(values), statistics.stdev(values)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["experiment_name"], []).append(row)

    aggregate: list[dict[str, Any]] = []
    for experiment_name, group_rows in sorted(grouped.items()):
        first = group_rows[0]
        out: dict[str, Any] = {
            "experiment_name": experiment_name,
            "experiment_label": first["experiment_label"],
            "group": first["group"],
            "n": len(group_rows),
        }
        for key in ["best_psnr", *[f"final_{metric}" for metric in METRICS]]:
            mean, std = mean_std([row.get(key) for row in group_rows])
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
        aggregate.append(out)
    return aggregate


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_markdown(path: Path, aggregate: list[dict[str, Any]]) -> None:
    headers = [
        "experiment",
        "n",
        "best_psnr",
        "final_psnr",
        "final_ssim",
        "pplx",
        "usage",
        "avg_pair_dist",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in aggregate:
        values = [
            row["experiment_name"],
            str(row["n"]),
            f"{format_float(row['best_psnr_mean'])} +/- {format_float(row['best_psnr_std'])}",
            f"{format_float(row['final_psnr_mean'])} +/- {format_float(row['final_psnr_std'])}",
            f"{format_float(row['final_ssim_mean'])} +/- {format_float(row['final_ssim_std'])}",
            f"{format_float(row['final_perplexity_mean'])} +/- {format_float(row['final_perplexity_std'])}",
            f"{format_float(row['final_usage_mean'])} +/- {format_float(row['final_usage_std'])}",
            f"{format_float(row['final_avg_pairwise_distance_mean'])} +/- {format_float(row['final_avg_pairwise_distance_std'])}",
        ]
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def aggregate(root: Path, suite_name: str) -> dict[str, Path | int]:
    summaries = collect_summaries(root, suite_name)
    if not summaries:
        raise FileNotFoundError(f"No summary files found under {root / suite_name}")

    rows = [flatten_summary(summary) for summary in summaries]
    aggregate = aggregate_rows(rows)
    out_dir = root / suite_name / "_aggregate"
    by_seed_path = out_dir / "summary_by_seed.csv"
    mean_std_path = out_dir / "summary_mean_std.csv"
    markdown_path = out_dir / "summary.md"

    write_csv(by_seed_path, rows, list(rows[0].keys()))
    write_csv(mean_std_path, aggregate, list(aggregate[0].keys()))
    write_markdown(markdown_path, aggregate)
    return {
        "count": len(rows),
        "by_seed": by_seed_path,
        "mean_std": mean_std_path,
        "markdown": markdown_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate VQ experiment summaries.")
    parser.add_argument("--root", type=Path, default=Path("./runs"))
    parser.add_argument("--suite-name", required=True)
    args = parser.parse_args()

    result = aggregate(args.root, args.suite_name)
    print(f"Aggregated {result['count']} runs")
    print(f"by_seed={result['by_seed']}")
    print(f"mean_std={result['mean_std']}")
    print(f"markdown={result['markdown']}")


if __name__ == "__main__":
    main()
