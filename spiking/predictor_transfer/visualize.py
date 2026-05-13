"""Visualize predictor transfer results as cross-benchmark heatmaps.

Supports both memorization transfer (run_mem_sim.py) and correctness
transfer (run_corr_sim.py) results.

Usage:
  # Memorization transfer (default): d_hat varies by source.
  uv run python spiking/predictor_transfer/visualize.py

  # Specific memorization predictors.
  uv run python spiking/predictor_transfer/visualize.py \
      --method min_k_plus_plus loss

  # Correctness transfer: c_hat varies by source.
  uv run python spiking/predictor_transfer/visualize.py \
      --mode corr

  # Specific correctness predictors.
  uv run python spiking/predictor_transfer/visualize.py \
      --mode corr --method llama_platt roberta

  # Average across all settings.
  uv run python spiking/predictor_transfer/visualize.py \
      --dose-group avg
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors

from spiking.config import (
    BENCHMARK_LABELS,
    CORR_LABELS,
    MEM_LABELS_TEX as MEM_LABELS,
    TEXT_WIDTH,
)

# Match the paper template with LaTeX rendering.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'

ANNOT_FONTSIZE = 7

ALL_BENCHMARKS = list(BENCHMARK_LABELS.keys())

# Plot settings for memorization and correctness transfer.
MODE_CONFIG = {
    'mem': {
        'predictor_col': 'mem_predictor',
        'labels': MEM_LABELS,
        'default_metric': 'ipw_rmse',
        'results_file': 'transfer_simulation_results.parquet',
        'title_prefix': 'Memorization Predictor Transfer',
    },
    'corr': {
        'predictor_col': 'corr_predictor',
        'labels': CORR_LABELS,
        'default_metric': 'imputation_rmse',
        'results_file': 'corr_transfer_simulation_results.parquet',
        'title_prefix': 'Correctness Predictor Transfer',
    },
}


def _split_benchmarks(benchmarks):
    """Split benchmarks into sources (all) and targets (no wikipedia).

    Wikipedia is source-only: it can supply predictors but is not a standard
    evaluation benchmark, so it never appears as a target.
    """
    sources = list(benchmarks)
    targets = [b for b in benchmarks if b != 'wikipedia']
    return sources, targets


def build_heatmap_matrix(df, sources, targets, predictor_col, method, dose_group,
                         metric='ipw_rmse'):
    """Build source x target matrix for one method.

    Returns (matrix, naive_row) where matrix is (n_src, n_tgt) and
    naive_row is shape (n_tgt,) with the naive RMSE per target benchmark.
    """
    subset = df[(df[predictor_col] == method) & (df['dose_group'] == dose_group)]
    n_src, n_tgt = len(sources), len(targets)
    matrix = np.full((n_src, n_tgt), np.nan)
    naive_row = np.full(n_tgt, np.nan)
    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            row = subset[(subset['source'] == src) & (subset['target'] == tgt)]
            if len(row) == 1:
                matrix[i, j] = row.iloc[0][metric]
                naive_row[j] = row.iloc[0]['naive_rmse']
    return matrix, naive_row


def build_average_heatmap_matrix(df, sources, targets, predictor_col, method,
                                 metric='ipw_rmse'):
    """Build source x target matrix averaged across all settings.

    Averages across all rows available for a given (source, target, method),
    e.g. low/mid/high random and easy/medium/hard correlated.
    """
    subset = df[df[predictor_col] == method]
    n_src, n_tgt = len(sources), len(targets)
    matrix = np.full((n_src, n_tgt), np.nan)
    naive_row = np.full(n_tgt, np.nan)

    for i, src in enumerate(sources):
        for j, tgt in enumerate(targets):
            row = subset[(subset['source'] == src) & (subset['target'] == tgt)]
            if len(row) > 0:
                matrix[i, j] = row[metric].mean()

    for j, tgt in enumerate(targets):
        row = df[df['target'] == tgt]
        if len(row) > 0:
            naive_vals = row[
                ['regime', 'dose_group', 'difficulty_bin', 'target', 'naive_rmse']
            ].drop_duplicates()['naive_rmse']
            if len(naive_vals) > 0:
                naive_row[j] = naive_vals.mean()

    return matrix, naive_row


DOSE_GROUPS = ['low', 'mid', 'high']


def _draw_panel(ax, matrix, naive_row, x_labels, y_labels_src, vmin, vmax,
                show_ylabels, scale=100):
    """Draw a single heatmap panel into ax.

    Naive row is placed at the top, followed by source benchmark rows.

    x_labels: target benchmark labels (columns).
    y_labels_src: source benchmark labels (rows, after the Naive row).
    """
    # Naive row at the top.
    extended = np.vstack([naive_row[np.newaxis, :], matrix])
    y_labels = ['Naive'] + y_labels_src

    annot_data = np.full(extended.shape, '', dtype=object)
    for r in range(extended.shape[0]):
        for c in range(extended.shape[1]):
            val = extended[r, c]
            if not np.isnan(val):
                annot_data[r, c] = f'{val * scale:.1f}'

    sns.heatmap(
        extended * scale,
        xticklabels=x_labels,
        yticklabels=y_labels if show_ylabels else False,
        annot=annot_data, fmt='',
        annot_kws={'size': ANNOT_FONTSIZE},
        cmap=colors.LinearSegmentedColormap.from_list(
            'Blues_trunc',
            plt.cm.Blues(np.linspace(0.25, 0.82, 256))
        ),
        vmin=vmin, vmax=vmax,
        cbar=False,
        linewidths=0.3,
        linecolor='white',
        ax=ax,
        square=True,
    )

    # Separator line below the Naive row.
    ax.axhline(y=1, color='black', linewidth=1.5)

    # Bold the diagonal, where source and target match.
    n_cols = extended.shape[1]
    for text_idx, text in enumerate(ax.texts):
        r, c = divmod(text_idx, n_cols)
        text.set_color('black')
        if r >= 1 and y_labels_src[r - 1] == x_labels[c]:
            text.set_fontweight('bold')

    ax.tick_params(axis='x', rotation=45, which='both')
    ax.tick_params(axis='y', rotation=0, which='both')


def plot_heatmaps(df, benchmarks, methods, dose_group, predictor_col,
                  method_labels, title_prefix, metric='ipw_rmse',
                  scale=100, out_path=None, vmin=None, vmax=None):
    """Plot one heatmap per method, arranged in a row."""
    sources, targets = _split_benchmarks(benchmarks)
    src_labels = [BENCHMARK_LABELS.get(b, b) for b in sources]
    tgt_labels = [BENCHMARK_LABELS.get(b, b) for b in targets]
    n_methods = len(methods)
    n_src = len(sources)
    n_rows_hm = n_src + 1  # naive plus source rows

    all_values, naive_rows = [], []
    for method in methods:
        if dose_group == 'avg':
            matrix, naive_row = build_average_heatmap_matrix(
                df, sources, targets, predictor_col, method, metric)
        else:
            matrix, naive_row = build_heatmap_matrix(
                df, sources, targets, predictor_col, method, dose_group, metric)
        all_values.append(matrix)
        naive_rows.append(naive_row)

    if vmin is None or vmax is None:
        all_vals_flat = np.concatenate([m.ravel() for m in all_values] + naive_rows)
        finite_vals = all_vals_flat[~np.isnan(all_vals_flat)]
        vmin = np.min(finite_vals) * scale
        vmax = np.max(finite_vals) * scale

    fig_w = TEXT_WIDTH
    cell_w = fig_w / (n_methods * len(targets))
    fig_h = cell_w * n_rows_hm
    fig, axes = plt.subplots(1, n_methods, figsize=(fig_w, fig_h), squeeze=False,
                             layout='constrained')
    axes = axes[0]

    for idx, (method, matrix, naive_row) in enumerate(
            zip(methods, all_values, naive_rows)):
        _draw_panel(axes[idx], matrix, naive_row, tgt_labels, src_labels,
                    vmin, vmax,
                    show_ylabels=(idx == 0),
                    scale=scale)
        axes[idx].set_title(method_labels.get(method, method),
                            fontsize=10, fontweight='bold', pad=4)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')

    fig.supxlabel('Target benchmark', fontsize=12, fontweight='bold')
    fig.supylabel('Source benchmark', fontsize=12, fontweight='bold')
    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close(fig)


def plot_heatmaps_all_doses(df, benchmarks, methods, predictor_col,
                            method_labels, metric='ipw_rmse',
                            scale=100, out_path=None):
    """Plot methods as rows, dose groups (low/mid/high) as columns."""
    sources, targets = _split_benchmarks(benchmarks)
    src_labels = [BENCHMARK_LABELS.get(b, b) for b in sources]
    tgt_labels = [BENCHMARK_LABELS.get(b, b) for b in targets]
    n_methods = len(methods)
    n_src = len(sources)
    n_doses = len(DOSE_GROUPS)
    n_rows_hm = n_src + 1  # naive plus source rows

    all_data = {}
    all_vals = []
    for method in methods:
        for dose in DOSE_GROUPS:
            matrix, naive_row = build_heatmap_matrix(df, sources, targets, predictor_col,
                                                     method, dose, metric)
            all_data[(method, dose)] = (matrix, naive_row)
            all_vals.extend([matrix.ravel(), naive_row])
    all_vals_flat = np.concatenate(all_vals)
    finite_vals = all_vals_flat[~np.isnan(all_vals_flat)]
    if finite_vals.size == 0:
        raise RuntimeError(
            f'No finite values to plot for methods={methods}, metric={metric}. '
            f'Input df has {len(df)} rows; sources={sources} targets={targets}. '
            'Check that run_mem_sim.py / run_corr_sim.py produced rows for the '
            'expected source/target benchmarks.'
        )
    vmin = np.min(finite_vals) * scale
    vmax = np.max(finite_vals) * scale

    fig_w = TEXT_WIDTH
    cell_w = fig_w / (n_doses * len(targets))
    fig_h = cell_w * n_rows_hm * n_methods

    fig = plt.figure(figsize=(fig_w, fig_h), layout='constrained')
    gs = fig.add_gridspec(n_methods, n_doses)
    axes = np.empty((n_methods, n_doses), dtype=object)
    for r in range(n_methods):
        for c in range(n_doses):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for row_idx, method in enumerate(methods):
        for col_idx, dose in enumerate(DOSE_GROUPS):
            ax = axes[row_idx, col_idx]
            matrix, naive_row = all_data[(method, dose)]

            _draw_panel(ax, matrix, naive_row, tgt_labels, src_labels,
                        vmin, vmax,
                        show_ylabels=(col_idx == 0),
                        scale=scale)

            if row_idx == 0:
                ax.set_title(f'{dose.capitalize()} dose',
                             fontsize=10, fontweight='bold', pad=4)

            method_label = method_labels.get(method, method)
            ax.set_ylabel(method_label if col_idx == 0 else '')
            ax.set_xlabel('')

    for r in range(n_methods - 1):
        for c in range(n_doses):
            axes[r, c].tick_params(labelbottom=False)

    fig.supxlabel('Target benchmark', fontsize=11, fontweight='bold')
    fig.supylabel('Source benchmark', fontsize=11, fontweight='bold')

    if out_path:
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved: {out_path}')
    plt.close(fig)


def _latex_escape(text):
    """Escape minimal LaTeX special chars for labels/captions."""
    replacements = {
        '\\': r'\textbackslash{}',
        '&': r'\&',
        '%': r'\%',
        '_': r'\_',
        '#': r'\#',
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    return out


def _format_cell(val, scale=100):
    if pd.isna(val):
        return '--'
    return f'{val * scale:.1f}'


def _build_latex_table_for_method(df, sources, targets, predictor_col, method,
                                  method_label, metric='ipw_rmse', scale=100):
    """Build one LaTeX table for a single method with low/mid/high blocks."""
    target_labels = [BENCHMARK_LABELS.get(t, t) for t in targets]
    source_labels = [BENCHMARK_LABELS.get(s, s) for s in sources]

    matrices = {}
    naive_rows = {}
    for dose in DOSE_GROUPS:
        matrix, naive_row = build_heatmap_matrix(
            df, sources, targets, predictor_col, method, dose, metric
        )
        matrices[dose] = matrix
        naive_rows[dose] = naive_row

    col_spec = 'l' + 'c' * (len(targets) * len(DOSE_GROUPS))
    lines = []
    lines.append(r'\begin{table*}[t]')
    lines.append(r'\centering')
    lines.append(r'\small')
    lines.append(r'\setlength{\tabcolsep}{4pt}')
    lines.append(r'\renewcommand{\arraystretch}{1.1}')
    lines.append(rf'\begin{{tabular}}{{{col_spec}}}')
    lines.append(r'\toprule')

    header1 = [r'\textbf{Source}']
    for dose in DOSE_GROUPS:
        header1.append(
            rf'\multicolumn{{{len(targets)}}}{{c}}{{\textbf{{{dose.capitalize()} dose}}}}'
        )
    lines.append(' & '.join(header1) + r' \\')

    cmidrules = []
    start = 2
    for _ in DOSE_GROUPS:
        end = start + len(targets) - 1
        cmidrules.append(rf'\cmidrule(lr){{{start}-{end}}}')
        start = end + 1
    lines.append(' '.join(cmidrules))

    header2 = ['']
    for _ in DOSE_GROUPS:
        header2.extend([rf'\textbf{{{_latex_escape(lbl)}}}' for lbl in target_labels])
    lines.append(' & '.join(header2) + r' \\')
    lines.append(r'\midrule')

    naive_cells = [r'\textbf{Naive}']
    for dose in DOSE_GROUPS:
        naive_cells.extend([_format_cell(v, scale=scale) for v in naive_rows[dose]])
    lines.append(' & '.join(naive_cells) + r' \\')
    lines.append(r'\midrule')

    for i, src in enumerate(sources):
        row_cells = [rf'\textbf{{{_latex_escape(source_labels[i])}}}']
        for dose in DOSE_GROUPS:
            row_cells.extend([
                _format_cell(matrices[dose][i, j], scale=scale)
                for j in range(len(targets))
            ])
        lines.append(' & '.join(row_cells) + r' \\')

    lines.append(r'\bottomrule')
    lines.append(r'\end{tabular}')
    lines.append(
        rf'\caption{{Cross-benchmark transfer RMSE (\(\times 100\)) for {_latex_escape(method_label)} across low, mid, and high dose settings. Rows denote source benchmarks and columns denote target benchmarks.}}'
    )
    lines.append(
        rf'\label{{tab:transfer_{method}_all_doses}}'
    )
    lines.append(r'\end{table*}')
    lines.append('')

    return '\n'.join(lines)


def write_latex_tables_all_doses(df, benchmarks, methods, predictor_col,
                                 method_labels, metric='ipw_rmse',
                                 scale=100, out_path=None):
    """Write LaTeX tables for the all-doses matrices.

    If multiple methods are provided, all method tables are concatenated
    into one .tex file.
    """
    sources, targets = _split_benchmarks(benchmarks)
    pieces = []
    for method in methods:
        method_label = method_labels.get(method, method)
        pieces.append(
            _build_latex_table_for_method(
                df=df,
                sources=sources,
                targets=targets,
                predictor_col=predictor_col,
                method=method,
                method_label=method_label,
                metric=metric,
                scale=scale,
            )
        )

    latex = '\n'.join(pieces)

    if out_path:
        out_path.write_text(latex)
        print(f'Saved: {out_path}')

    return latex


def main():
    parser = argparse.ArgumentParser(
        description='Visualize predictor transfer as cross-benchmark heatmaps')
    parser.add_argument('--mode', default='mem', choices=['mem', 'corr'],
                        help='Transfer mode: mem (d_hat varies) or corr (c_hat varies)')
    parser.add_argument('--benchmarks', nargs='+', default=None,
                        choices=ALL_BENCHMARKS,
                        help='Benchmarks to include (default: all)')
    parser.add_argument('--method', nargs='+', default=None,
                        help='Methods to plot (default: all available)')
    parser.add_argument('--dose-group', default='high',
                        choices=['low', 'mid', 'high', 'all', 'avg'],
                        help='Dose group to show, "all" for low/mid/high side by side, or "avg" for average across all settings (default: high)')
    parser.add_argument('--metric', default=None,
                        help='Metric to plot (default: ipw_rmse)')
    parser.add_argument('--results-file', default=None,
                        help='Path to results parquet (default: auto-detect)')
    args = parser.parse_args()

    cfg = MODE_CONFIG[args.mode]
    predictor_col = cfg['predictor_col']
    method_labels = cfg['labels']
    metric = args.metric or cfg['default_metric']

    if args.results_file:
        results_path = Path(args.results_file)
    else:
        results_path = RESULTS_DIR / cfg['results_file']
    df = pd.read_parquet(results_path)

    benchmarks = args.benchmarks or ALL_BENCHMARKS
    methods = args.method or [
        m for m in method_labels if m in df[predictor_col].unique()
    ]

    available_src = set(df['source'].unique())
    available_tgt = set(df['target'].unique())
    benchmarks = [b for b in benchmarks
                  if b in available_src or b in available_tgt]

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    prefix = args.mode
    sources, targets = _split_benchmarks(benchmarks)

    if args.dose_group == 'all':
        plot_heatmaps_all_doses(
            df, benchmarks, methods, predictor_col,
            method_labels, metric,
            out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_all_doses.pdf',
        )
        write_latex_tables_all_doses(
            df, benchmarks, methods, predictor_col,
            method_labels, metric,
            out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_all_doses.tex',
        )

        for method in methods:
            plot_heatmaps_all_doses(
                df, benchmarks, [method], predictor_col,
                method_labels, metric,
                out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_{method}_all_doses.pdf',
            )
            write_latex_tables_all_doses(
                df, benchmarks, [method], predictor_col,
                method_labels, metric,
                out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_{method}_all_doses.tex',
            )
    else:
        all_vals = []
        for method in methods:
            if args.dose_group == 'avg':
                matrix, naive_row = build_average_heatmap_matrix(
                    df, sources, targets, predictor_col, method, metric)
            else:
                matrix, naive_row = build_heatmap_matrix(
                    df, sources, targets, predictor_col, method, args.dose_group, metric)
            all_vals.extend([matrix.ravel(), naive_row])

        all_vals_flat = np.concatenate(all_vals)
        finite_vals = all_vals_flat[~np.isnan(all_vals_flat)]
        global_vmin = np.min(finite_vals) * 100
        global_vmax = np.max(finite_vals) * 100

        suffix = 'average' if args.dose_group == 'avg' else args.dose_group

        plot_heatmaps(
            df, benchmarks, methods, args.dose_group, predictor_col,
            method_labels, cfg['title_prefix'], metric,
            out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_{suffix}.pdf',
            vmin=global_vmin, vmax=global_vmax,
        )
        for method in methods:
            plot_heatmaps(
                df, benchmarks, [method], args.dose_group, predictor_col,
                method_labels, cfg['title_prefix'], metric,
                out_path=FIGURES_DIR / f'{prefix}_transfer_heatmap_{method}_{suffix}.pdf',
                vmin=global_vmin, vmax=global_vmax,
            )


if __name__ == '__main__':
    main()
