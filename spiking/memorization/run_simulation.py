"""Run simulations with cached memorization d_hat predictions.

For each benchmark/model/attack, loads the d_hat array (from run.py),
runs Monte Carlo simulation under random and correlated contamination,
and reports RMSE for the naive estimator and IPW with each MIA predictor.

Outputs one table per benchmark. Benchmarks without confidence scores
(e.g. PopQA) are skipped for correlated contamination.

No GPU needed. Runs in minutes on CPU.

Usage:
  uv run python spiking/memorization/run_simulation.py
  uv run python spiking/memorization/run_simulation.py --benchmark mmlu
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from spiking.config import (
    BENCHMARKS,
    BENCHMARK_LABELS as BENCHMARK_LABELS_SHORT,
    DOSE_GROUPS,
    MEM_PREDICTORS as ALL_PREDICTORS,
    MODELS,
)
from hubble.results import load_sim_item_pool
from hubble.simulation import (
    DIFFICULTY_BINS,
    SamplerConfig,
    TestSet,
    ipw_estimator,
    naive_estimator,
    sample_test_set,
)

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
PAPER_DIR = Path(__file__).resolve().parent.parent
EXP50_DIR = PAPER_DIR / 'data_generation' / 'results'

ATTACK_LABELS = {
    'loss': 'IPW (LOSS)',
    'zlib': 'IPW (Zlib)',
    'min_k': 'IPW (Min-K%)',
    'min_k_plus_plus': 'IPW (Min-K%++)',
    'reference': 'IPW (Reference)',
    'final_layer_linear': 'IPW (HS mean)',
    'final_layer_last_token_linear': 'IPW (HS last)',
}

def run_sim(pool, regime, dose_group, n, gamma, n_replicates, seed,
            difficulty_bin='hard'):
    """Run Monte Carlo simulation, return RMSE and mean accuracy for naive and IPW."""
    cfg = SamplerConfig(
        regime=regime, n=n, gamma=gamma,
        dose_group=dose_group, difficulty_bin=difficulty_bin,
    )

    # IPW and naive do not use c_hat, but TestSet expects it.
    dummy_c_hat = np.zeros(pool.n_items)

    rng = np.random.default_rng(seed)
    naive_errors = []
    ipw_errors = []
    naive_accs = []
    ipw_accs = []
    gt_accs = []
    balanced_accs = []

    for _ in range(n_replicates):
        indices = sample_test_set(pool, cfg, rng)
        ts = TestSet(
            y_observed=pool.y_observed[indices],
            y_clean=pool.y_clean[indices],
            d_hat=pool.d_hat[indices],
            c_hat=dummy_c_hat[indices],
            indices=indices,
        )
        gt = ts.ground_truth
        naive_est = naive_estimator(ts)
        ipw_est = ipw_estimator(ts)
        naive_errors.append((naive_est - gt) ** 2)
        ipw_errors.append((ipw_est - gt) ** 2)
        naive_accs.append(naive_est)
        ipw_accs.append(ipw_est)
        gt_accs.append(gt)

        # Balanced accuracy of the d_hat predictor.
        d_true = pool.duplicates[indices] > 0  # contamination label
        d_pred = ts.d_hat >= 0.5
        if len(np.unique(d_true)) < 2:
            balanced_accs.append(np.nan)
        else:
            balanced_accs.append(balanced_accuracy_score(d_true, d_pred))

    return {
        'naive_rmse': np.sqrt(np.mean(naive_errors)),
        'ipw_rmse': np.sqrt(np.mean(ipw_errors)),
        'ground_truth_acc': np.mean(gt_accs),
        'naive_acc': np.mean(naive_accs),
        'ipw_acc': np.mean(ipw_accs),
        'balanced_acc': np.nanmean(balanced_accs),
    }


def _fmt_rmse(val):
    return f'{val * 100:.1f}'


def _fmt_acc(val):
    return f'{val * 100:.1f}'


def format_benchmark_table(benchmark, results_df, has_correlated):
    """Format one benchmark's results as an HTML table.

    Rows: Naive, then IPW with each attack.
    Columns: Random (Low/Mid/High dose) and optionally Correlated (Easy/Medium/Hard).
    Each condition shows RMSE, Weighted Acc, and Balanced Acc (for IPW rows).
    """
    lines = [f'### {BENCHMARK_LABELS_SHORT[benchmark]}', '', '<table>']

    condition_labels = ['Low', 'Mid', 'High']
    if has_correlated:
        condition_labels += ['Easy', 'Medium', 'Hard']

    # Top-level column groups.
    header1 = '<tr><th rowspan="2">Estimator</th>'
    header1 += f'<th colspan="{3}">Random</th>'
    if has_correlated:
        header1 += f'<th colspan="{3}">Correlated (high dose)</th>'
    header1 += '</tr>'
    lines.append(header1)

    # Per-condition columns.
    header2 = '<tr>'
    for label in condition_labels:
        header2 += f'<th>{label}</th>'
    header2 += '</tr>'
    lines.append(header2)

    def _get_subset(regime, sub_key, attack=None):
        if regime == 'random':
            mask = (
                (results_df['regime'] == 'random')
                & (results_df['dose_group'] == sub_key)
            )
        else:
            mask = (
                (results_df['regime'] == 'correlated')
                & (results_df['difficulty_bin'] == sub_key)
            )
        if attack is not None:
            mask = mask & (results_df['attack'] == attack)
        return results_df[mask]

    def _iter_conditions():
        """Yield (regime, sub_key) for each column."""
        for dg in ['low', 'mid', 'high']:
            yield 'random', dg
        if has_correlated:
            for db in DIFFICULTY_BINS:
                yield 'correlated', db

    # Naive row: RMSE and weighted accuracy.
    naive_row = '<tr><td>Naive</td>'
    for regime, sub_key in _iter_conditions():
        subset = _get_subset(regime, sub_key)
        if len(subset) > 0:
            r = subset.iloc[0]
            naive_row += f'<td>{_fmt_rmse(r["naive_rmse"])} ({_fmt_acc(r["naive_acc"])})</td>'
        else:
            naive_row += '<td>—</td>'
    naive_row += '</tr>'
    lines.append(naive_row)

    # IPW rows: RMSE, weighted accuracy, and balanced accuracy.
    unique_attacks = results_df['attack'].unique()
    for attack in unique_attacks:
        row = f'<tr><td>{ATTACK_LABELS[attack]}</td>'
        for regime, sub_key in _iter_conditions():
            subset = _get_subset(regime, sub_key, attack)
            if len(subset) > 0:
                r = subset.iloc[0]
                row += (f'<td>{_fmt_rmse(r["ipw_rmse"])} '
                        f'({_fmt_acc(r["ipw_acc"])}) '
                        f'[bal={_fmt_acc(r["balanced_acc"])}]</td>')
            else:
                row += '<td>—</td>'
        row += '</tr>'
        lines.append(row)

    lines.append('</table>')
    return '\n'.join(lines)


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500, help='Test set size')
    parser.add_argument('--gamma', type=float, default=0.3, help='Contamination rate')
    parser.add_argument('--n-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='8b-500b')
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=list(BENCHMARKS),
                        help='Single benchmark (default: all)')
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    all_rows = []
    tables = []

    for benchmark in benchmarks:
        print(f'\n=== {benchmark} / {args.model} ===')
        sim_pool = load_sim_item_pool(EXP50_DIR, benchmark, args.model)
        has_confidence = sim_pool.confidence is not None
        print(f'  Items: {sim_pool.n_items} total, '
              f'{len(sim_pool.clean_idx)} clean, '
              f'{len(sim_pool.contaminated_idx)} contaminated')
        if not has_confidence:
            print('  No confidence scores — skipping correlated regime')

        d_hat_path = RESULTS_DIR / benchmark / f'd_hat_{args.model}.npz'
        d_hat_data = np.load(d_hat_path)
        available_attacks = [a for a in ALL_PREDICTORS if a in d_hat_data]

        benchmark_rows = []

        # Random regime: sweep dose groups.
        for dose_group in DOSE_GROUPS:
            for attack in available_attacks:
                sim_pool.d_hat = d_hat_data[attack]
                result = run_sim(
                    sim_pool, 'random', dose_group, args.n, args.gamma,
                    args.n_replicates, args.seed,
                )
                row = {
                    'benchmark': benchmark, 'model': args.model,
                    'regime': 'random', 'attack': attack,
                    'dose_group': dose_group, 'difficulty_bin': None,
                    **result,
                }
                benchmark_rows.append(row)
                print(f'  random / {dose_group} / {attack}: '
                      f'naive={result["naive_rmse"]*100:.1f}pp (acc={result["naive_acc"]*100:.1f}); '
                      f'ipw={result["ipw_rmse"]*100:.1f}pp (acc={result["ipw_acc"]*100:.1f}); '
                      f'bal_acc={result["balanced_acc"]*100:.1f}; '
                      f'gt={result["ground_truth_acc"]*100:.1f}')

        # Correlated regime: high dose across difficulty bins.
        if has_confidence:
            for difficulty_bin in DIFFICULTY_BINS:
                for attack in available_attacks:
                    sim_pool.d_hat = d_hat_data[attack]
                    result = run_sim(
                        sim_pool, 'correlated', 'high', args.n, args.gamma,
                        args.n_replicates, args.seed,
                        difficulty_bin=difficulty_bin,
                    )
                    row = {
                        'benchmark': benchmark, 'model': args.model,
                        'regime': 'correlated', 'attack': attack,
                        'dose_group': 'high', 'difficulty_bin': difficulty_bin,
                        **result,
                    }
                    benchmark_rows.append(row)
                    print(f'  correlated / {difficulty_bin} / {attack}: '
                          f'naive={result["naive_rmse"]*100:.1f}pp (acc={result["naive_acc"]*100:.1f}); '
                          f'ipw={result["ipw_rmse"]*100:.1f}pp (acc={result["ipw_acc"]*100:.1f}); '
                          f'bal_acc={result["balanced_acc"]*100:.1f}; '
                          f'gt={result["ground_truth_acc"]*100:.1f}')

        bm_df = pd.DataFrame(benchmark_rows)
        all_rows.extend(benchmark_rows)
        tables.append(format_benchmark_table(benchmark, bm_df, has_confidence))

    # Save combined results.
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_DIR / 'simulation_results.csv', index=False)
    results_df.to_parquet(RESULTS_DIR / 'simulation_results.parquet')

    # Write tables.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full_output = '\n\n'.join(tables)
    table_path = FIGURES_DIR / 'simulation_tables.md'
    table_path.write_text(full_output)
    print(f'\n{full_output}')
    print(f'\nSaved: {table_path}')


if __name__ == '__main__':
    main()
