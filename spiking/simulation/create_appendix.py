"""Create appendix figures for simulation phase diagrams.

Generates two full-page figures from saved parquet results:
  1. Random contamination (benchmarks x 3 dose levels)
  2. Correlated contamination (benchmarks x 3 difficulty bins)

Usage:
  uv run python src/spiking/simulation/create_appendix.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spiking.config import BENCHMARK_LABELS, TEXT_WIDTH
from hubble.simulation import ESTIMATORS

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"

MODEL = "8b-500b"


def plot_phase_panel(ax, df, est_names, cmap, title):
    best = df.loc[df.groupby(["alpha_m", "bias_c"])["mse"].idxmin()]
    pivot = best.pivot(index="bias_c", columns="alpha_m", values="estimator")
    est_to_int = {name: i for i, name in enumerate(est_names)}
    pivot_int = pivot.map(lambda x: est_to_int.get(x, -1)).astype(float)
    ax.imshow(
        pivot_int.values, cmap=cmap,
        vmin=-0.5, vmax=len(est_names) - 0.5, aspect="auto",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.max(), pivot.index.min(),
        ],
    )
    ax.set_title(title)


def create_appendix_figure(benchmarks, regime, col_keys, col_labels, save_path):
    """Create a full-page figure: rows = benchmarks, cols = dose/difficulty levels."""
    n_rows = len(benchmarks)
    n_cols = len(col_keys)
    est_names = list(ESTIMATORS.keys())
    cmap = plt.colormaps.get_cmap("Set2").resampled(len(est_names))

    fig = plt.figure(figsize=(TEXT_WIDTH, TEXT_WIDTH * 0.24 * n_rows))
    gs = fig.add_gridspec(n_rows, n_cols + 1,
                          width_ratios=[1] * n_cols + [0.06],
                          wspace=0.12, hspace=0.12)

    axes = np.empty((n_rows, n_cols), dtype=object)

    for r, bench in enumerate(benchmarks):
        for c, key in enumerate(col_keys):
            path = RESULTS_DIR / bench / MODEL / \
                f"probe_sweep_{regime}_{key}.parquet"
            df = pd.read_parquet(path)

            share_kw = {}
            if c > 0 and axes[r, 0] is not None:
                share_kw = dict(sharey=axes[r, 0])
            ax = fig.add_subplot(gs[r, c], **share_kw)

            title = col_labels[c] if r == 0 else ""
            plot_phase_panel(ax, df, est_names, cmap, title)
            ax.set_xlabel("")
            ax.set_ylabel("")
            axes[r, c] = ax

    # Sync x-axis limits across all panels.
    xlim = axes[0, 0].get_xlim()
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xlim(xlim)

    # Hide interior tick labels.
    for r in range(n_rows):
        for c in range(n_cols):
            if r < n_rows - 1:
                axes[r, c].tick_params(labelbottom=False)
            if c > 0:
                axes[r, c].tick_params(labelleft=False)

        # Keep y-ticks sparse and readable.
        ymin, ymax = axes[r, 0].get_ylim()
        hi = max(abs(ymin), abs(ymax))
        for step in [0.05, 0.1, 0.2, 0.5]:
            ticks = np.arange(0, hi + step, step)
            ticks = ticks[ticks <= hi + 1e-9]
            if len(ticks) <= 3:
                break
        axes[r, 0].set_yticks(ticks)
        axes[r, 0].set_yticklabels([f"{t:.2f}" for t in ticks])

    # Benchmark labels on the left column.
    for r, bench in enumerate(benchmarks):
        axes[r, 0].set_ylabel(BENCHMARK_LABELS[bench] + "\n")

    fig.canvas.draw()
    ax_left = axes[0, 0].get_position().x0
    fig.text(ax_left * 0.45, 0.5,
             r"Correctness predictor $|\mathrm{bias}|$ ($\rightarrow$ better)",
             va="center", ha="center", rotation=90, fontsize=8)
    fig.supxlabel(r"Memorization predictor AUROC ($\rightarrow$ better)")

    # Colorbar.
    cax = fig.add_subplot(gs[:, n_cols])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(-0.5, len(est_names) - 0.5))
    cbar = fig.colorbar(sm, cax=cax, ticks=range(len(est_names)))
    cbar.ax.set_yticklabels(est_names)

    fig.subplots_adjust(right=0.86, bottom=0.07 + 0.02 * (5 - n_rows))
    stem = save_path.with_suffix("")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"Saved {stem}.{{pdf,png}}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    random_benchmarks = ["mmlu", "hellaswag", "winogrande_mcq", "piqa", "popqa"]
    correlated_benchmarks = ["mmlu", "hellaswag", "winogrande_mcq", "piqa"]

    # Random contamination by dose.
    create_appendix_figure(
        random_benchmarks,
        regime="random",
        col_keys=["low", "mid", "high"],
        col_labels=["Low", "Mid", "High"],
        save_path=FIGURES_DIR / "appendix_random.png",
    )

    # Correlated contamination by difficulty bin.
    create_appendix_figure(
        correlated_benchmarks,
        regime="correlated",
        col_keys=["easy", "medium", "hard"],
        col_labels=["Easy", "Medium", "Hard"],
        save_path=FIGURES_DIR / "appendix_correlated.png",
    )

    print("Done.")


if __name__ == "__main__":
    main()
