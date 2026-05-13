"""Run simulations with cached d_hat and c_hat predictions.

For each benchmark/model, loads d_hat and c_hat arrays, runs Monte Carlo
simulation under random and correlated contamination, and reports RMSE for
Naive, IPW, Imputation, and Combined.

No GPU needed. Runs in minutes on CPU.

Usage:
  uv run python spiking/correctness/run_simulation.py
  uv run python spiking/correctness/run_simulation.py --benchmark mmlu

  # Include cached Pythia predictions.
  uv run python spiking/correctness/run_simulation.py --pythia-size 6.9b
  uv run python spiking/correctness/run_simulation.py --pythia-size 6.9b --perturbed-labels
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import (
    BENCHMARKS,
    BENCHMARK_LABELS as BENCHMARK_LABELS_SHORT,
    CORR_LABELS,
    CORR_PREDICTORS_FULL as CORR_PREDICTORS,
    DOSE_GROUPS,
    MEM_LABELS,
    MEM_PREDICTORS,
    MODELS,
    PYTHIA_SIZES,
)
from hubble.results import load_sim_item_pool
from hubble.simulation import (
    DIFFICULTY_BINS,
    SamplerConfig,
    TestSet,
    combined_estimator,
    imputation_estimator,
    ipw_estimator,
    naive_estimator,
    sample_test_set,
)

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
PAPER_DIR = Path(__file__).resolve().parent.parent
DATA_RESULTS = PAPER_DIR / 'data_generation' / 'results'
MEM_RESULTS = PAPER_DIR / 'memorization' / 'results'

def run_sim(pool, regime, dose_group, n, gamma, n_replicates, seed,
            difficulty_bin='hard'):
    """Run Monte Carlo simulation, return RMSE for all four estimators."""
    cfg = SamplerConfig(
        regime=regime, n=n, gamma=gamma,
        dose_group=dose_group, difficulty_bin=difficulty_bin,
    )

    rng = np.random.default_rng(seed)
    errors = {name: [] for name in ['naive', 'ipw', 'imputation', 'combined']}

    for _ in range(n_replicates):
        indices = sample_test_set(pool, cfg, rng)
        ts = TestSet(
            y_observed=pool.y_observed[indices],
            y_clean=pool.y_clean[indices],
            d_hat=pool.d_hat[indices],
            c_hat=pool.c_hat[indices],
            indices=indices,
        )
        gt = ts.ground_truth
        errors['naive'].append((naive_estimator(ts) - gt) ** 2)
        errors['ipw'].append((ipw_estimator(ts) - gt) ** 2)
        errors['imputation'].append((imputation_estimator(ts) - gt) ** 2)
        errors['combined'].append((combined_estimator(ts) - gt) ** 2)

    return {f'{name}_rmse': np.sqrt(np.mean(errs))
            for name, errs in errors.items()}


def format_benchmark_table(benchmark, results_df, has_correlated, mem_predictor):
    """Format one benchmark's results as an HTML table.

    Rows: Naive, IPW, then Imputation and Combined for each corr predictor.
    Columns: Random (Low/Mid/High dose) and optionally Correlated (Easy/Medium/Hard).
    """
    lines = [f'### {BENCHMARK_LABELS_SHORT[benchmark]}', '', '<table>']

    # Top-level column groups.
    header1 = '<tr><th rowspan="2">Estimator</th>'
    header1 += '<th colspan="3">Random</th>'
    if has_correlated:
        header1 += '<th colspan="3">Correlated (high dose)</th>'
    header1 += '</tr>'
    lines.append(header1)

    # Per-condition columns.
    header2 = '<tr>'
    header2 += '<th>Low</th><th>Mid</th><th>High</th>'
    if has_correlated:
        header2 += '<th>Easy</th><th>Medium</th><th>Hard</th>'
    header2 += '</tr>'
    lines.append(header2)

    def lookup_rmse(estimator_col, corr_predictor, regime, sub_key):
        """Look up a formatted RMSE cell."""
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
        if corr_predictor is not None:
            mask = mask & (results_df['corr_predictor'] == corr_predictor)
        subset = results_df[mask]
        if len(subset) == 0:
            return '—'
        return f'{subset.iloc[0][estimator_col] * 100:.1f}'

    def make_row(label, estimator_col, corr_predictor=None):
        row = f'<tr><td>{label}</td>'
        for dg in ['low', 'mid', 'high']:
            row += f'<td>{lookup_rmse(estimator_col, corr_predictor, "random", dg)}</td>'
        if has_correlated:
            for db in DIFFICULTY_BINS:
                row += f'<td>{lookup_rmse(estimator_col, corr_predictor, "correlated", db)}</td>'
        row += '</tr>'
        return row

    # Naive and IPW are shared across correctness predictors.
    lines.append(make_row('Naive', 'naive_rmse'))
    lines.append(make_row(f'IPW ({MEM_LABELS[mem_predictor]})', 'ipw_rmse'))

    unique_corr = results_df['corr_predictor'].unique()
    for corr_predictor in unique_corr:
        lines.append(make_row(
            f'Imputation ({CORR_LABELS[corr_predictor]})',
            'imputation_rmse', corr_predictor))

    for corr_predictor in unique_corr:
        lines.append(make_row(
            f'Combined ({CORR_LABELS[corr_predictor]})',
            'combined_rmse', corr_predictor))

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
    parser.add_argument('--mem-predictor', type=str, default='min_k_plus_plus',
                        choices=MEM_PREDICTORS,
                        help='Memorization predictor for d_hat (default: min_k_plus_plus)')
    parser.add_argument('--pythia-size', type=str, default=None,
                        choices=PYTHIA_SIZES,
                        help='Include pythia_platt predictor from this Pythia size (e.g. 6.9b)')
    parser.add_argument('--perturbed-labels', action='store_true',
                        help='Load pythia c_hat from perturbed_labels results (default: standard_labels)')
    parser.add_argument('--question-only', action='store_true',
                        help='Load question-only RoBERTa c_hat (from run_roberta.py --question-only)')
    args = parser.parse_args()

    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    all_rows = []
    tables = []

    for benchmark in benchmarks:
        print(f'\n=== {benchmark} / {args.model} ===')
        sim_pool = load_sim_item_pool(DATA_RESULTS, benchmark, args.model)
        has_confidence = sim_pool.confidence is not None
        print(f'  Items: {sim_pool.n_items} total, '
              f'{len(sim_pool.clean_idx)} clean, '
              f'{len(sim_pool.contaminated_idx)} contaminated')
        if not has_confidence:
            print('  No confidence scores — skipping correlated regime')

        # Load cached d_hat.
        d_hat_path = MEM_RESULTS / benchmark / f'd_hat_{args.model}.npz'
        if not d_hat_path.exists():
            print(f'  [SKIP] No d_hat at {d_hat_path}')
            continue
        d_hat_data = np.load(d_hat_path)
        if args.mem_predictor not in d_hat_data:
            print(f'  [SKIP] {args.mem_predictor} not found in {d_hat_path}')
            continue
        sim_pool.d_hat = d_hat_data[args.mem_predictor]

        # Merge all c_hat files for the requested label variant.
        label_suffix = 'perturbed_labels' if args.perturbed_labels else 'standard_labels'
        labeled_results_dir = RESULTS_DIR / label_suffix

        qonly_suffix = '_qonly' if args.question_only else ''

        c_hat_all = {}
        for filename in [
            f'c_hat_llama_{args.model}.npz',
            f'c_hat_roberta_{args.model}{qonly_suffix}.npz',
            f'c_hat_{args.model}.npz',
        ]:
            c_hat_path = labeled_results_dir / benchmark / filename
            if c_hat_path.exists():
                data = np.load(c_hat_path)
                for key in data:
                    if key not in c_hat_all:
                        c_hat_all[key] = data[key]

        # Add Pythia c_hat when requested.
        if args.pythia_size:
            pythia_label = f'pythia-{args.pythia_size}'
            pythia_path = labeled_results_dir / benchmark / f'c_hat_pythia_{pythia_label}_{args.model}.npz'
            if pythia_path.exists():
                data = np.load(pythia_path)
                for key in data:
                    if key not in c_hat_all:
                        c_hat_all[key] = data[key]
            else:
                print(f'  [WARN] Pythia c_hat not found: {pythia_path}')

        available_corr = [c for c in CORR_PREDICTORS if c in c_hat_all]
        if not available_corr:
            print(f'  [SKIP] No c_hat files found for {benchmark}')
            continue

        print(f'  d_hat: {args.mem_predictor}, c_hat predictors: {available_corr}')

        benchmark_rows = []

        # Random regime: dose groups x correctness predictors.
        for dose_group in DOSE_GROUPS:
            for corr_predictor in available_corr:
                sim_pool.c_hat = c_hat_all[corr_predictor]
                result = run_sim(
                    sim_pool, 'random', dose_group, args.n, args.gamma,
                    args.n_replicates, args.seed,
                )
                row = {
                    'benchmark': benchmark, 'model': args.model,
                    'regime': 'random', 'dose_group': dose_group,
                    'difficulty_bin': None,
                    'mem_predictor': args.mem_predictor, 'corr_predictor': corr_predictor,
                    **result,
                }
                benchmark_rows.append(row)
                print(f'  random / {dose_group} / {corr_predictor}: '
                      f'naive={result["naive_rmse"]*100:.1f}pp; '
                      f'ipw={result["ipw_rmse"]*100:.1f}pp; '
                      f'impute={result["imputation_rmse"]*100:.1f}pp; '
                      f'combined={result["combined_rmse"]*100:.1f}pp')

        # Correlated regime: high dose x difficulty bins x correctness predictors.
        if has_confidence:
            for difficulty_bin in DIFFICULTY_BINS:
                for corr_predictor in available_corr:
                    sim_pool.c_hat = c_hat_all[corr_predictor]
                    result = run_sim(
                        sim_pool, 'correlated', 'high', args.n, args.gamma,
                        args.n_replicates, args.seed,
                        difficulty_bin=difficulty_bin,
                    )
                    row = {
                        'benchmark': benchmark, 'model': args.model,
                        'regime': 'correlated', 'dose_group': 'high',
                        'difficulty_bin': difficulty_bin,
                        'mem_predictor': args.mem_predictor, 'corr_predictor': corr_predictor,
                        **result,
                    }
                    benchmark_rows.append(row)
                    print(f'  correlated / {difficulty_bin} / {corr_predictor}: '
                          f'naive={result["naive_rmse"]*100:.1f}pp; '
                          f'ipw={result["ipw_rmse"]*100:.1f}pp; '
                          f'impute={result["imputation_rmse"]*100:.1f}pp; '
                          f'combined={result["combined_rmse"]*100:.1f}pp')

        bm_df = pd.DataFrame(benchmark_rows)
        all_rows.extend(benchmark_rows)
        tables.append(format_benchmark_table(
            benchmark, bm_df, has_confidence, args.mem_predictor))

    # Keep filenames aligned with optional inputs.
    out_suffix = ''
    if args.pythia_size:
        out_suffix += f'_pythia-{args.pythia_size}'
    if args.perturbed_labels:
        out_suffix += '_perturbed'
    if args.question_only:
        out_suffix += '_qonly'

    # Save combined results.
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_DIR / f'simulation_results{out_suffix}.csv', index=False)
    results_df.to_parquet(RESULTS_DIR / f'simulation_results{out_suffix}.parquet')

    # Write tables.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full_output = '\n\n'.join(tables)
    table_path = FIGURES_DIR / f'simulation_tables{out_suffix}.md'
    table_path.write_text(full_output)
    print(f'\n{full_output}')
    print(f'\nSaved: {table_path}')


if __name__ == '__main__':
    main()
