"""Reporting helpers for correctness experiments."""

from __future__ import annotations

from spiking.config import BENCHMARK_LABELS as BENCHMARK_LABELS_LONG
from hubble.simulation import DIFFICULTY_BINS


BIAS_COLS = ['bias_easy', 'bias_medium', 'bias_hard', 'bias_diff_all']
BIAS_HEADERS = ['Easy', 'Med', 'Hard', 'All']


def metrics_to_row(brier, auroc, bal_acc, bias, variance, n_cal, n_sim):
    """Convert metric dictionaries to one flat table row."""
    row = {
        'brier_clean': brier['clean'],
        'brier_contaminated': brier['contaminated'],
        'brier_all': brier['all'],
        'brier_low': brier['low'],
        'brier_mid': brier['mid'],
        'brier_high': brier['high'],
        'auroc_clean': auroc['clean'],
        'auroc_contaminated': auroc['contaminated'],
        'auroc_all': auroc['all'],
        'auroc_low': auroc['low'],
        'auroc_mid': auroc['mid'],
        'auroc_high': auroc['high'],
        'bal_acc_clean': bal_acc['clean'],
        'bal_acc_contaminated': bal_acc['contaminated'],
        'bal_acc_all': bal_acc['all'],
        'bal_acc_low': bal_acc['low'],
        'bal_acc_mid': bal_acc['mid'],
        'bal_acc_high': bal_acc['high'],
        'bias_clean': bias['clean'],
        'bias_contaminated': bias['contaminated'],
        'bias_all': bias['all'],
        'bias_diff_all': bias['diff_all'],
        'bias_low': bias['low'],
        'bias_mid': bias['mid'],
        'bias_high': bias['high'],
        'var_clean': variance['clean'],
        'var_contaminated': variance['contaminated'],
        'var_all': variance['all'],
        'var_diff_all': variance['diff_all'],
        'var_low': variance['low'],
        'var_mid': variance['mid'],
        'var_high': variance['high'],
    }
    for b in DIFFICULTY_BINS:
        row[f'brier_{b}'] = brier[b]
        row[f'auroc_{b}'] = auroc[b]
        row[f'bal_acc_{b}'] = bal_acc[b]
        row[f'bias_{b}'] = bias[b]
        row[f'var_{b}'] = variance[b]
    row['n_cal'] = n_cal
    row['n_sim'] = n_sim
    return row


def print_full_metrics(predictor_name, brier, auroc, bal_acc, bias, variance):
    """Print one-line summary for a predictor."""
    print(f'  {predictor_name:<20s}  '
          f'brier: clean={brier["clean"]:.4f} contam={brier["contaminated"]:.4f} all={brier["all"]:.4f}  '
          f'auroc: low={auroc["low"]:.4f} mid={auroc["mid"]:.4f} high={auroc["high"]:.4f} all={auroc["all"]:.4f}  '
          f'bal_acc: low={bal_acc["low"]:.4f} mid={bal_acc["mid"]:.4f} high={bal_acc["high"]:.4f} all={bal_acc["all"]:.4f}  '
          f'bias: low={bias["low"]:.4f} mid={bias["mid"]:.4f} high={bias["high"]:.4f} all={bias["all"]:.4f}  '
          f'bias(diff): easy={bias["easy"]:.4f} med={bias["medium"]:.4f} hard={bias["hard"]:.4f} all={bias["diff_all"]:.4f}  '
          f'var(diff): easy={variance["easy"]:.4f} med={variance["medium"]:.4f} hard={variance["hard"]:.4f} all={variance["diff_all"]:.4f}')


def format_quality_table(quality_df):
    """Format results as markdown tables grouped by benchmark."""
    lines = []
    for benchmark, bm_group in quality_df.groupby('benchmark'):
        lines.append(f'\n### {benchmark}\n')
        header_cols = [
            'Predictor', 'Model',
            'Brier Clean', 'Brier Contam', 'Brier All',
            'AUROC Low', 'AUROC Mid', 'AUROC High',
            'AUROC Clean', 'AUROC Contam', 'AUROC All',
            'BalAcc Low', 'BalAcc Mid', 'BalAcc High',
            'BalAcc Clean', 'BalAcc Contam', 'BalAcc All',
            'Bias Low', 'Bias Mid', 'Bias High',
            'Bias Clean', 'Bias Contam', 'Bias All',
            'Bias Easy', 'Bias Med', 'Bias Hard', 'Bias Diff All',
            'Var Easy', 'Var Med', 'Var Hard', 'Var Diff All',
        ]
        lines.append('| ' + ' | '.join(header_cols) + ' |')
        lines.append('|' + '|'.join(['---'] * len(header_cols)) + '|')
        for _, row in bm_group.iterrows():
            lines.append(
                f'| {row["predictor"]:<20s} '
                f'| {row["model"]:<8s} '
                f'| {row["brier_clean"]:.4f}      '
                f'| {row["brier_contaminated"]:.4f}       '
                f'| {row["brier_all"]:.4f}   '
                f'| {row["auroc_low"]:.4f}    '
                f'| {row["auroc_mid"]:.4f}    '
                f'| {row["auroc_high"]:.4f}     '
                f'| {row["auroc_clean"]:.4f}      '
                f'| {row["auroc_contaminated"]:.4f}       '
                f'| {row["auroc_all"]:.4f}    '
                f'| {row["bal_acc_low"]:.4f}     '
                f'| {row["bal_acc_mid"]:.4f}    '
                f'| {row["bal_acc_high"]:.4f}      '
                f'| {row["bal_acc_clean"]:.4f}       '
                f'| {row["bal_acc_contaminated"]:.4f}        '
                f'| {row["bal_acc_all"]:.4f}     '
                f'| {row["bias_low"]:.4f}   '
                f'| {row["bias_mid"]:.4f}   '
                f'| {row["bias_high"]:.4f}    '
                f'| {row["bias_clean"]:.4f}     '
                f'| {row["bias_contaminated"]:.4f}      '
                f'| {row["bias_all"]:.4f}   '
                f'| {row["bias_easy"]:.4f}   '
                f'| {row["bias_medium"]:.4f}  '
                f'| {row["bias_hard"]:.4f}   '
                f'| {row["bias_diff_all"]:.4f}   '
                f'| {row["var_easy"]:.4f}   '
                f'| {row["var_medium"]:.4f}  '
                f'| {row["var_hard"]:.4f}   '
                f'| {row["var_diff_all"]:.4f}   |'
            )
    return '\n'.join(lines)


def _bold_best(val, best_val):
    """Format value, bolding if it equals the column best."""
    s = f'{val:.4f}'
    return rf'\textbf{{{s}}}' if abs(val - best_val) < 1e-6 else s


def format_bias_latex_table(quality_df):
    """LaTeX table showing absolute bias by difficulty bin."""
    benchmarks = [b for b in BENCHMARK_LABELS_LONG if b in quality_df['benchmark'].unique()]
    predictors = quality_df['predictor'].unique()

    best = {}
    for bench in benchmarks:
        bdf = quality_df[quality_df['benchmark'] == bench]
        for col in BIAS_COLS:
            vals = bdf[col].dropna().abs()
            best[(bench, col)] = bdf.loc[vals.idxmin(), col] if len(vals) > 0 else float('nan')

    n = len(benchmarks)
    lines = [
        r'\begin{table*}[t]',
        r'\centering',
        r'\small',
        r'\setlength{\tabcolsep}{3.5pt}',
        r'\begin{tabular}{l' + 'cccc' * n + '}',
        r'\toprule',
    ]
    h1 = r'\textbf{Predictor}'
    for bench in benchmarks:
        h1 += rf' & \multicolumn{{4}}{{c}}{{\textbf{{{BENCHMARK_LABELS_LONG[bench]}}}}}'
    lines.append(h1 + r' \\')
    lines.append(' '.join(
        rf'\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}' for i in range(n)
    ))
    h2 = ''
    for _ in range(n):
        for h in BIAS_HEADERS:
            h2 += rf' & \textbf{{{h}}}'
    lines.append(h2 + r' \\')
    lines.append(r'\midrule')
    for predictor in predictors:
        row = predictor.replace('_', r'\_')
        for bench in benchmarks:
            sub = quality_df[(quality_df['benchmark'] == bench) & (quality_df['predictor'] == predictor)]
            if sub.empty:
                row += ' & -- & -- & -- & --'
                continue
            r = sub.iloc[0]
            for col in BIAS_COLS:
                row += ' & ' + _bold_best(r[col], best[(bench, col)])
        lines.append(row + r' \\')
    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\caption{Correctness predictor absolute bias ($|\text{mean error}|$) by item difficulty bin. '
        r'Lower values indicate less biased predictions.}',
        r'\label{tab:corr_abs_bias}',
        r'\end{table*}',
    ]
    return '\n'.join(lines)


__all__ = [
    'format_quality_table',
    'format_bias_latex_table',
    'metrics_to_row',
    'print_full_metrics',
    'BIAS_COLS',
    'BIAS_HEADERS',
]
