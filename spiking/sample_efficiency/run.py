"""Plot sample efficiency results as a 2×3 grid.

Rows = dose level (high top, mid bottom).
Columns = benchmarks (winogrande_mcq, mmlu, popqa by default).

Loads pre-computed results from results/sample_efficiency_{dose}.parquet.

Usage:
  uv run python src/spiking/sample_efficiency/run.py
  uv run python src/spiking/sample_efficiency/run.py --dose-groups high mid low
  uv run python src/spiking/sample_efficiency/run.py --benchmarks winogrande_mcq mmlu popqa
  uv run python src/spiking/sample_efficiency/run.py --estimators ipw combined correctness
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

plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'

DEFAULT_BENCHMARKS = ['winogrande_mcq', 'mmlu', 'popqa']
DEFAULT_DOSE_GROUPS = ['high', 'mid']

Y_AXIS_CFG = {
    'winogrande_mcq': {
        'ylim': (2.0, 27.0),
        'yticks': [10, 20],
    },
    'mmlu': {
        'ylim': (3.0, 17.5),
        'yticks': [5, 10, 15],
    },
    'popqa': {
        'ylim': (2.0, 27.0),
        'yticks': [10, 20],
    },
}

ESTIMATOR_DEFS = {
    'ipw':         ('ipw_rmse',         'o', '-',  'IPW (Min-K++)'),
    'imputation': ('correctness_rmse', 'D', '-',  'Imputation'),
    'clean_only':  ('clean_only_rmse',  's', '--', 'Clean-only'),
    'naive':       ('naive_rmse',       None, ':', 'Naive (no correction)'),
}

ALL_ESTIMATORS = list(ESTIMATOR_DEFS.keys())
DEFAULT_ESTIMATORS = ['ipw', 'imputation', 'clean_only', 'naive']


def load_dose_df(dose_group):
    path = RESULTS_DIR / f'sample_efficiency_{dose_group}.parquet'
    if not path.exists():
        raise FileNotFoundError(f'Results not found: {path}')
    return pd.read_parquet(path)


def apply_y_axis_style(ax, benchmark):
    cfg = Y_AXIS_CFG.get(benchmark)
    if cfg is None:
        return
    ax.set_ylim(*cfg['ylim'])
    ax.set_yticks(cfg['yticks'])


def plot_panel(ax, bm_df, benchmark, estimators=DEFAULT_ESTIMATORS,
               show_legend=False, show_xlabel=False, title=None):
    x = bm_df['n_cal_target'].values

    for est_key in estimators:
        col, marker, ls, label = ESTIMATOR_DEFS[est_key]
        if col not in bm_df.columns or bm_df[col].isna().all():
            continue

        y = bm_df[col].values * 100
        if marker is None:
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

    apply_y_axis_style(ax, benchmark)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Pull tick labels closer to the axes.
    ax.tick_params(axis='y', pad=2)
    ax.tick_params(axis='x', pad=1)

    # Keep panels tall enough for the compact paper layout.
    ax.set_box_aspect(0.95)

    ax.grid(False)
    ax.set_axisbelow(True)

    if not show_xlabel:
        ax.tick_params(labelbottom=False)

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)

    if show_legend:
        ax.legend(
            loc='upper right',
            fontsize=7,
            handlelength=1.0,
            handletextpad=0.3,
            borderpad=0.2,
            labelspacing=0.2,
            markerscale=0.8,
        )


def main():
    parser = argparse.ArgumentParser(
        description='Plot sample efficiency results as a dose × benchmark grid')
    parser.add_argument('--benchmarks', nargs='+', default=DEFAULT_BENCHMARKS,
                        choices=list(BENCHMARK_LABELS.keys()))
    parser.add_argument('--dose-groups', nargs='+', default=DEFAULT_DOSE_GROUPS,
                        choices=['high', 'mid', 'low'])
    parser.add_argument('--estimators', nargs='+', default=DEFAULT_ESTIMATORS,
                        choices=ALL_ESTIMATORS)
    parser.add_argument('--out', type=str, default=None,
                        help='Output PDF path (default: figures/sample_efficiency_grid.pdf)')
    args = parser.parse_args()

    benchmarks = args.benchmarks
    dose_groups = args.dose_groups
    n_rows = len(dose_groups)
    n_cols = len(benchmarks)

    dose_data = {dg: load_dose_df(dg) for dg in dose_groups}

    # Give each dose row enough vertical space.
    fig_w = TEXT_WIDTH
    fig_h = 2.7 * n_rows

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharey='col',
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
                    fontsize=9, color='0.5'
                )
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                apply_y_axis_style(ax, benchmark)
                ax.set_box_aspect(0.95)
                continue

            plot_panel(
                ax, bm_df, benchmark,
                estimators=args.estimators,
                show_legend=(row_idx == 0 and col_idx == 0),
                show_xlabel=is_bottom_row,
                title=BENCHMARK_LABELS.get(benchmark, benchmark) if row_idx == 0 else None,
            )

    if n_rows > 1:
        for row_idx, dose_group in enumerate(dose_groups):
            row_top = 1.0 - row_idx / n_rows
            row_bot = 1.0 - (row_idx + 1) / n_rows
            row_mid = (row_top + row_bot) / 2
            fig.text(
                0.03, row_mid,
                DOSE_LABELS.get(dose_group, dose_group),
                ha='left', va='center',
                fontsize=12,
                fontweight='bold',
                rotation=90,
            )

    fig.supylabel('RMSE (pp)', fontsize=12, fontweight='bold', x=0.03)
    fig.supxlabel('Calibration set size', fontsize=12, fontweight='bold', y=0.1)

    # Manual spacing is steadier than constrained layout here.
    fig.subplots_adjust(
        left=0.12,
        right=0.995,
        bottom=0.22,
        top=0.84,
        wspace=0.28,
        hspace=0.18,
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(FIGURES_DIR / 'sample_efficiency_grid.pdf')
    fig.savefig(out_path, bbox_inches='tight', pad_inches=0.01)
    print(f'Saved: {out_path}')
    plt.close(fig)


if __name__ == '__main__':
    main()
