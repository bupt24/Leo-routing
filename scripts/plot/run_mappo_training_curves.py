#!/usr/bin/env python3
"""Plot separate Vanilla MAPPO training curves from a metrics CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, PercentFormatter  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "remote_sensing_mappo" / "20260625_155330"
DEFAULT_METRICS_CSV = DEFAULT_RUN_DIR / "mappo_training_metrics.csv"

REQUIRED_COLUMNS = {
    "episode",
    "steps",
    "success_rate",
    "avg_delay_success_ms",
    "avg_cls_delay_success_ms",
    "avg_reward_all",
}
X_AXIS_LABELS = {
    "episode": "Episode",
    "cumulative_steps": "Cumulative steps",
    "cumulative_tasks": "Cumulative tasks",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split MAPPO training metrics into four standalone PNG curves."
    )
    parser.add_argument(
        "--metrics-csv",
        default=str(DEFAULT_METRICS_CSV),
        help="Path to mappo_training_metrics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        help="Directory for output PNGs. Defaults to the metrics CSV parent.",
    )
    parser.add_argument("--prefix", default="mappo_300", help="Output filename prefix.")
    parser.add_argument(
        "--x-axis",
        choices=sorted(X_AXIS_LABELS),
        default="episode",
        help="Column to use as the x-axis.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    return parser.parse_args()


def read_numeric_column(row: dict[str, str], column: str, row_number: int) -> float:
    value = row.get(column, "")
    if value == "":
        raise ValueError(f"Missing value for column {column!r} at CSV row {row_number}.")
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(
            f"Invalid numeric value for column {column!r} at CSV row {row_number}: {value!r}"
        ) from exc


def load_metrics(metrics_csv: Path, x_axis: str) -> dict[str, list[float]]:
    if not metrics_csv.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_csv}")

    with metrics_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        fieldnames = set(reader.fieldnames or [])
        required_columns = set(REQUIRED_COLUMNS)
        required_columns.add(x_axis)
        missing = sorted(required_columns - fieldnames)
        if missing:
            raise ValueError(f"Metrics CSV is missing required columns: {', '.join(missing)}")

        metrics = {column: [] for column in required_columns}
        for row_number, row in enumerate(reader, start=2):
            for column in required_columns:
                metrics[column].append(read_numeric_column(row, column, row_number))

    if not metrics["episode"]:
        raise ValueError(f"Metrics CSV has no data rows: {metrics_csv}")
    return metrics


def finish_plot(ax: plt.Axes, output_path: Path, dpi: int, x_label: str) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.set_xlabel(x_label)
    ax.figure.tight_layout()
    ax.figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(ax.figure)


def set_reward_axis(ax: plt.Axes, rewards: list[float]) -> None:
    y_min = math.floor(min(rewards) / 0.2) * 0.2
    y_max = 0.0 if max(rewards) < 0.0 else math.ceil(max(rewards) / 0.2) * 0.2
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def set_delay_axis(ax: plt.Axes, *series: list[float], y_min: float = 0.0) -> None:
    values = [value for values in series for value in values]
    y_max = math.ceil(max(values) / 20.0) * 20.0
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_locator(MultipleLocator(20.0))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))


def plot_reward(metrics: dict[str, list[float]], x_axis: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics[x_axis], metrics["avg_reward_all"], color="#1f77b4", linewidth=1.8)
    ax.set_title("MAPPO Training Reward")
    ax.set_ylabel("Average reward")
    set_reward_axis(ax, metrics["avg_reward_all"])
    finish_plot(ax, output_path, dpi, X_AXIS_LABELS[x_axis])


def plot_delay(metrics: dict[str, list[float]], x_axis: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(
        metrics[x_axis],
        metrics["avg_delay_success_ms"],
        color="#d62728",
        linewidth=1.8,
        label="End-to-end delay",
    )
    ax.plot(
        metrics[x_axis],
        metrics["avg_cls_delay_success_ms"],
        color="#2ca02c",
        linewidth=1.8,
        label="CLS route delay",
    )
    ax.set_title("MAPPO Training Delay")
    ax.set_ylabel("Delay (ms)")
    ax.legend(frameon=False)
    set_delay_axis(ax, metrics["avg_delay_success_ms"], metrics["avg_cls_delay_success_ms"], y_min=0.0)
    finish_plot(ax, output_path, dpi, X_AXIS_LABELS[x_axis])


def plot_success_rate(metrics: dict[str, list[float]], x_axis: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics[x_axis], metrics["success_rate"], color="#9467bd", linewidth=1.8)
    ax.set_title("MAPPO Training Success Rate")
    ax.set_ylabel("Success rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(0.0, min(1.05, max(metrics["success_rate"]) * 1.08))
    finish_plot(ax, output_path, dpi, X_AXIS_LABELS[x_axis])


def plot_steps(metrics: dict[str, list[float]], x_axis: str, output_path: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(metrics[x_axis], metrics["steps"], color="#ff7f0e", linewidth=1.8)
    ax.set_title("MAPPO Training Steps")
    ax.set_ylabel("Steps")
    finish_plot(ax, output_path, dpi, X_AXIS_LABELS[x_axis])


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else metrics_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_metrics(metrics_csv, args.x_axis)
    outputs = {
        "reward": output_dir / f"{args.prefix}_reward_curve.png",
        "delay": output_dir / f"{args.prefix}_delay_curve.png",
        "success_rate": output_dir / f"{args.prefix}_success_rate_curve.png",
        "steps": output_dir / f"{args.prefix}_steps_curve.png",
    }

    plot_reward(metrics, args.x_axis, outputs["reward"], args.dpi)
    plot_delay(metrics, args.x_axis, outputs["delay"], args.dpi)
    plot_success_rate(metrics, args.x_axis, outputs["success_rate"], args.dpi)
    plot_steps(metrics, args.x_axis, outputs["steps"], args.dpi)

    print(f"Read metrics: {metrics_csv}")
    print(f"X-axis: {args.x_axis}")
    for name, output_path in outputs.items():
        print(f"Wrote {name}: {output_path}")


if __name__ == "__main__":
    main()
