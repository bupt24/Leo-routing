#!/usr/bin/env python3
"""Plot the averaged Random end-to-end delay curve from an avg-window CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MultipleLocator  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = REPO_ROOT / "outputs" / "remote_sensing_random_episode_curve" / "20260611_234115"
DEFAULT_CSV = DEFAULT_RUN_DIR / "random_episode_end_to_end_delay_avg100.csv"
DEFAULT_OUTPUT = DEFAULT_RUN_DIR / "random_episode_end_to_end_delay_avg100_y0.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot Random avg100 end-to-end delay with fixed y-axis ticks.")
    parser.add_argument("--csv", default=str(DEFAULT_CSV), help="Input avg-window CSV.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output PNG path.")
    parser.add_argument("--dpi", type=int, default=180, help="Output PNG DPI.")
    return parser.parse_args()


def load_rows(csv_path: Path) -> tuple[list[float], list[float]]:
    episodes: list[float] = []
    delays: list[float] = []
    with csv_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        required = {"episode_midpoint", "avg_delay_ms"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            episodes.append(float(row["episode_midpoint"]))
            delays.append(float(row["avg_delay_ms"]))
    if not episodes:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return episodes, delays


def plot_curve(episodes: list[float], delays: list[float], output_path: Path, dpi: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        episodes,
        delays,
        color="#1f77b4",
        marker="^",
        markersize=6,
        linewidth=1.6,
        label="Random",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel("Average End-to-End Delay (ms)")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 120)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    ax.grid(True, axis="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    episodes, delays = load_rows(csv_path)
    plot_curve(episodes, delays, output_path, args.dpi)
    print(f"Read CSV: {csv_path}")
    print(f"Wrote plot: {output_path}")


if __name__ == "__main__":
    main()
