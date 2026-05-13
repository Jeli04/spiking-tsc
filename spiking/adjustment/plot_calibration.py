"""Plot contaminated, adjusted, and ground-truth accuracy bars.

For each estimator (Naive, IPW, Imputation, Combined) and benchmark,
draws overlapping bars for:
  - Contaminated accuracy (naive_mean, before adjustment)
  - Estimated adjusted accuracy (estimator output)
  - Ground-truth clean accuracy

Usage:
  uv run python spiking/adjustment/plot_calibration.py
  uv run python spiking/adjustment/plot_calibration.py --suffix all_8b-500b_min_k_plus_plus_n500_g0.3
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from spiking.config import BENCHMARK_LABELS

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'

ESTIMATORS = ['naive', 'ipw', 'imputation', 'combined']
ESTIMATOR_LABELS = {
    'naive': 'Naive',
    'ipw': 'IPW',
    'imputation': 'Imputation',
    'combined': 'Combined',
}

# X-axis categories: (regime, key column, key value, label).
RANDOM_CATS = [
    ('random', 'dose_group', 'low', 'Low'),
    ('random', 'dose_group', 'mid', 'Mid'),
    ('random', 'dose_group', 'high', 'High'),
]
CORRELATED_CATS = [
    ('correlated', 'difficulty_bin', 'easy', 'Easy'),
    ('correlated', 'difficulty_bin', 'medium', 'Medium'),
    ('correlated', 'difficulty_bin', 'hard', 'Hard'),
]


def lookup_val(df, col, regime, key_col, key_val):
    """Look up one value from the results dataframe."""
    mask = df['regime'] == regime
    mask = mask & (df[key_col] == key_val)
    subset = df[mask]
    if len(subset) == 0:
        return np.nan
    return subset.iloc[0][col]


def plot_calibration(results_df, corr_predictor, suffix):
    """Create the combined calibration figure."""
    # Use one correctness predictor for the whole figure.
    df = results_df[results_df['corr_predictor'] == corr_predictor].copy()

    benchmarks = [b for b in BENCHMARK_LABELS if b in df['benchmark'].unique()]
    n_benchmarks = len(benchmarks)
    if n_benchmarks == 0:
        print('No benchmarks found in results.')
        return

    colors = sns.color_palette('Set2', 3)
    bar_cfg = [
        # Draw widest bars first.
        ('Contaminated', 0.7, colors[0], 0.7),
        ('Estimated', 0.45, colors[1], 0.85),
        ('Ground truth', 0.2, colors[2], 1.0),
    ]

    fig, axes = plt.subplots(
        len(ESTIMATORS), n_benchmarks,
        figsize=(3.2 * n_benchmarks, 3.2 * len(ESTIMATORS)),
        sharey='row', squeeze=False,
    )

    for row, estimator in enumerate(ESTIMATORS):
        for col, benchmark in enumerate(benchmarks):
            ax = axes[row, col]
            bm_df = df[df['benchmark'] == benchmark]

            # Include correlated bins only when present.
            has_correlated = any(bm_df['regime'] == 'correlated')
            cats = list(RANDOM_CATS)
            if has_correlated:
                cats += CORRELATED_CATS

            # Leave a small gap between random and correlated groups.
            x_pos = []
            for i, cat in enumerate(cats):
                if i < len(RANDOM_CATS):
                    x_pos.append(i)
                else:
                    x_pos.append(i + 0.5)
            x_pos = np.array(x_pos)
            x_labels = [c[3] for c in cats]

            # Collect values for each category.
            contaminated = []
            estimated = []
            ground_truth = []
            for regime, key_col, key_val, _ in cats:
                contaminated.append(
                    lookup_val(bm_df, 'naive_mean', regime, key_col, key_val) * 100)
                estimated.append(
                    lookup_val(bm_df, f'{estimator}_mean', regime, key_col, key_val) * 100)
                ground_truth.append(
                    lookup_val(bm_df, 'ground_truth_mean', regime, key_col, key_val) * 100)

            # Draw overlapping bars.
            for vals, (label, width, color, alpha) in zip(
                    [contaminated, estimated, ground_truth], bar_cfg):
                ax.bar(x_pos, vals, width=width, color=color, alpha=alpha,
                       label=label, edgecolor='white', linewidth=0.5)

            ax.set_xticks(x_pos)
            ax.set_xticklabels(x_labels, fontsize=8)
            ax.tick_params(axis='y', labelsize=8)

            # Group labels.
            if has_correlated:
                random_center = np.mean(x_pos[:3])
                corr_center = np.mean(x_pos[3:])
                trans = ax.get_xaxis_transform()
                for center, glabel in [(random_center, 'Random'),
                                       (corr_center, 'Correlated')]:
                    ax.text(center, -0.15, glabel, transform=trans,
                            ha='center', fontsize=7, color='gray')

            # Panel labels.
            if row == 0:
                ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=10)
            if col == 0:
                ax.set_ylabel(f'{ESTIMATOR_LABELS[estimator]}\nAccuracy (%)',
                              fontsize=9)

    # Single legend at the top.
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center',
               bbox_to_anchor=(0.5, 1.02), ncol=3, fontsize=9,
               frameon=False)

    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURES_DIR / f'calibration_{suffix}.png'
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


def main():
    parser = argparse.ArgumentParser(
        description='Plot calibration bars: contaminated vs estimated vs ground truth.')
    parser.add_argument('--suffix', type=str,
                        default='all_8b-500b_min_k_plus_plus_n500_g0.3',
                        help='Filename suffix matching run_simulation output')
    parser.add_argument('--corr-predictor', type=str, default='platt',
                        help='Correctness predictor to use (default: platt)')
    args = parser.parse_args()

    csv_path = RESULTS_DIR / f'simulation_results_{args.suffix}.csv'
    if not csv_path.exists():
        print(f'Results not found: {csv_path}')
        return

    results_df = pd.read_csv(csv_path)
    plot_calibration(results_df, args.corr_predictor, args.suffix)


if __name__ == '__main__':
    main()
