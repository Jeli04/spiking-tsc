"""Plot sample efficiency results as a 2×3 grid.

Rows = dose level (high top, mid bottom).
Columns = benchmarks (winogrande_mcq, mmlu, popqa by default).

Loads pre-computed results from results/sample_efficiency_{dose}.parquet.

Usage:
  uv run python spiking/sample_efficiency/plot.py
  uv run python spiking/sample_efficiency/plot.py --dose-groups high mid low
  uv run python spiking/sample_efficiency/plot.py --benchmarks winogrande_mcq mmlu popqa
  uv run python spiking/sample_efficiency/plot.py --estimators ipw combined correctness
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from spiking.config import (
    BENCHMARK_LABELS,
    DOSE_LABELS,
    SAMPLE_EFF_COLORS as COLORS,
    TEXT_WIDTH,
)

# Match the paper template with LaTeX rendering.
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

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'

DEFAULT_BENCHMARKS = ['winogrande_mcq', 'mmlu', 'popqa']
DEFAULT_DOSE_GROUPS = ['high', 'mid']

# Estimator key -> (column, marker, linestyle, label).
ESTIMATOR_DEFS = {
    'ipw': ('ipw_rmse', 'o', '-', 'IPW (Min-K++)'),
    'imputation': ('correctness_rmse', 'D', '-', 'Imputation (Llama 3.1)'),
    'clean_only': ('clean_only_rmse', 's', '-', 'Clean-only'),
    'naive': ('naive_rmse', None, ':', 'Naive'),
}

ALL_ESTIMATORS = list(ESTIMATOR_DEFS.keys())
DEFAULT_ESTIMATORS = ['ipw', 'imputation', 'clean_only', 'naive']


def load_dose_df(dose_group):
    """Load results parquet for a given dose group."""
    path = RESULTS_DIR / f'sample_efficiency_{dose_group}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Results not found: {path}')
    return pd.read_parquet(path)


def plot_panel(ax, bm_df, benchmark, estimators=DEFAULT_ESTIMATORS,
               show_legend=False, show_xlabel=False, title=None):
    """Plot one sample-efficiency panel onto ax."""
    x = bm_df['n_cal_target'].values

    for est_key in estimators:
        col, marker, ls, label = ESTIMATOR_DEFS[est_key]
        if col not in bm_df.columns or bm_df[col].isna().all():
            continue

        y = bm_df[col].values * 100
        if marker is None:
            # Naive is a horizontal reference line.
            ax.axhline(
                y[0],
                color=COLORS[est_key],
                linewidth=1.6,
                linestyle=ls,
                label=label,
            )
        else:
            ax.plot(
                x, y, marker + ls,
                label=label,
                color=COLORS[est_key],
                markeredgewidth=0.4,
                markeredgecolor='white',
                clip_on=False,
            )

    ax.set_xscale('log', base=2)
    ax.set_xticks(x)
    ax.set_xticklabels(x)

    # Remove top/right spines.
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.grid(False)
    ax.set_axisbelow(True)

    if not show_xlabel:
        ax.tick_params(labelbottom=False)
    if title:
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
    if show_legend:
        ax.legend(
            loc='upper right',
            fontsize=5.0,
            handlelength=1.2,
            handletextpad=0.4,
            borderpad=0.3,
            labelspacing=0.3,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Plot sample efficiency results as a dose × benchmark grid')
    parser.add_argument('--benchmarks', nargs='+', default=DEFAULT_BENCHMARKS,
                        choices=list(BENCHMARK_LABELS.keys()),
                        help='Benchmarks for columns (default: winogrande_mcq mmlu popqa)')
    parser.add_argument('--dose-groups', nargs='+', default=DEFAULT_DOSE_GROUPS,
                        choices=['high', 'mid', 'low'],
                        help='Dose groups for rows, top to bottom (default: high mid)')
    parser.add_argument('--estimators', nargs='+', default=DEFAULT_ESTIMATORS,
                        choices=ALL_ESTIMATORS,
                        help='Estimators to plot (default: all)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output PDF path (default: figures/sample_efficiency_grid.pdf)')
    args = parser.parse_args()

    benchmarks = args.benchmarks
    dose_groups = args.dose_groups
    n_rows = len(dose_groups)
    n_cols = len(benchmarks)

    dose_data = {}
    for dg in dose_groups:
        dose_data[dg] = load_dose_df(dg)

    fig_w = TEXT_WIDTH * n_cols / 3
    fig_h = TEXT_WIDTH * 0.30 * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        layout='constrained',
        sharey='col',
        gridspec_kw={'height_ratios': [0.1] * n_rows},
    )

    for row_idx, dose_group in enumerate(dose_groups):
        df = dose_data[dose_group]
        is_bottom_row = row_idx == n_rows - 1

        for col_idx, benchmark in enumerate(benchmarks):
            ax = axes[row_idx][col_idx]
            bm_df = df[df['benchmark'] == benchmark].sort_values('n_cal_target')

            if len(bm_df) == 0:
                ax.text(
                    0.5, 0.5, f'No data\n{benchmark}',
                    ha='center', va='center',
                    transform=ax.transAxes,
                    fontsize=8, color='0.5'
                )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                continue

            plot_panel(
                ax, bm_df, benchmark,
                estimators=args.estimators,
                show_legend=(row_idx == 0 and col_idx == 0),
                show_xlabel=is_bottom_row,
                title=BENCHMARK_LABELS.get(benchmark, benchmark) if row_idx == 0 else None,
            )

    # Center the naive line within each benchmark column.
    for col_idx, benchmark in enumerate(benchmarks):
        ax = axes[0][col_idx]  # shared by the column
        naive_vals = []
        for dg in dose_groups:
            bm_df = dose_data[dg]
            bm_df = bm_df[bm_df['benchmark'] == benchmark]
            if len(bm_df) > 0 and 'naive_rmse' in bm_df.columns:
                naive_vals.append(bm_df['naive_rmse'].iloc[0] * 100)
        if naive_vals:
            naive_val = sum(naive_vals) / len(naive_vals)
            ymin, ymax = ax.get_ylim()
            max_dist = max(abs(ymax - naive_val), abs(naive_val - ymin))
            ax.set_ylim(naive_val - max_dist, naive_val + max_dist)

    # Row labels, when there is more than one dose group.
    if n_rows > 1:
        for row_idx, dose_group in enumerate(dose_groups):
            row_top = 1.0 - row_idx / n_rows
            row_bot = 1.0 - (row_idx + 1) / n_rows
            row_mid = (row_top + row_bot) / 2
            fig.text(
                0.03, row_mid,
                DOSE_LABELS.get(dose_group, dose_group),
                ha='left', va='center',
                fontsize=8, fontweight='bold',
                rotation=90,
            )

    fig.supylabel('RMSE (pp)', fontsize=9, fontweight='bold')
    fig.supxlabel('Calibration set size', fontsize=10, fontweight='bold')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(FIGURES_DIR / 'sample_efficiency_grid.pdf')
    fig.savefig(out_path, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
