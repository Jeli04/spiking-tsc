"""Run adjustment simulations with cached d_hat and c_hat predictions.

For each benchmark, loads d_hat and c_hat arrays, runs Monte Carlo simulation
under random and correlated contamination, and reports RMSE for Naive, IPW,
Imputation, and Combined.

Default: Min-K%++ for d_hat, Llama Platt + RoBERTa for c_hat.

No GPU needed. Runs in minutes on CPU.

Usage:
  uv run python spiking/adjustment/run_simulation.py
  uv run python spiking/adjustment/run_simulation.py --benchmark mmlu
  uv run python spiking/adjustment/run_simulation.py --benchmark popqa
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import (
    BENCHMARKS,
    BENCHMARK_LABELS as BENCHMARK_LABELS_SHORT,
    CORR_LABELS,
    CORR_PREDICTORS_DEFAULT as CORR_PREDICTORS,
    DOSE_GROUPS,
    MEM_LABELS,
    MEM_PREDICTORS,
    MODELS,
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
EXP50_DIR = DATA_RESULTS
EXP53_DIR = PAPER_DIR / 'memorization' / 'results'
EXP54_DIR = PAPER_DIR / 'correctness' / 'results'

def run_sim(pool, regime, dose_group, n, gamma, n_replicates, seed,
            difficulty_bin='hard'):
    """Run Monte Carlo simulation, return RMSE for all four estimators."""
    cfg = SamplerConfig(
        regime=regime, n=n, gamma=gamma,
        dose_group=dose_group, difficulty_bin=difficulty_bin,
    )

    rng = np.random.default_rng(seed)
    est_fns = [
        ('naive', naive_estimator),
        ('ipw', ipw_estimator),
        ('imputation', imputation_estimator),
        ('combined', combined_estimator),
    ]
    errors = {name: [] for name, _ in est_fns}
    estimates = {name: [] for name, _ in est_fns}
    ground_truths = []

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
        ground_truths.append(gt)
        for name, fn in est_fns:
            est = fn(ts)
            estimates[name].append(est)
            errors[name].append((est - gt) ** 2)

    result = {f'{name}_rmse': np.sqrt(np.mean(errs))
              for name, errs in errors.items()}
    result.update({f'{name}_mean': np.mean(ests)
                   for name, ests in estimates.items()})
    result['ground_truth_mean'] = np.mean(ground_truths)
    return result


def format_benchmark_table(benchmark, results_df, has_correlated, mem_predictor):
    """Format one benchmark's results as a markdown table.

    Rows: Naive, IPW, then Imputation and Combined for each corr predictor.
    Columns: Random (Low/Mid/High dose) and optionally Correlated (Easy/Medium/Hard).
    Values are RMSE × 100 (percentage points).
    """
    lines = [f'### {BENCHMARK_LABELS_SHORT[benchmark]}', '']

    # Header.
    cols = ['Estimator', 'Low', 'Mid', 'High']
    if has_correlated:
        cols += ['Easy', 'Medium', 'Hard']
    lines.append('| ' + ' | '.join(cols) + ' |')
    lines.append('|' + '|'.join(['---'] * len(cols)) + '|')

    def lookup_rmse(estimator_col, corr_predictor, regime, sub_key):
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
        cells = [label]
        for dg in ['low', 'mid', 'high']:
            cells.append(lookup_rmse(estimator_col, corr_predictor, 'random', dg))
        if has_correlated:
            for db in DIFFICULTY_BINS:
                cells.append(lookup_rmse(estimator_col, corr_predictor, 'correlated', db))
        return '| ' + ' | '.join(cells) + ' |'

    lines.append(make_row('Naive', 'naive_rmse'))
    lines.append(make_row(f'IPW ({MEM_LABELS[mem_predictor]})', 'ipw_rmse'))

    unique_corr = results_df['corr_predictor'].unique()
    for corr_predictor in unique_corr:
        lines.append(make_row(
            f'Imputation ({CORR_LABELS.get(corr_predictor, corr_predictor)})',
            'imputation_rmse', corr_predictor))

    for corr_predictor in unique_corr:
        lines.append(make_row(
            f'Combined ({CORR_LABELS.get(corr_predictor, corr_predictor)})',
            'combined_rmse', corr_predictor))

    return '\n'.join(lines)


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Run adjustment simulation with real d_hat and c_hat predictions.')
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
    parser.add_argument('--labels', type=str, default='standard_labels',
                        choices=['standard_labels', 'perturbed_labels'],
                        help='Which c_hat label variant to use (default: standard_labels)')
    parser.add_argument('--question-only', action='store_true',
                        help='Load question-only c_hat results (from run_llama/roberta/pythia/qwen --question-only)')
    args = parser.parse_args()

    qonly_suffix = '_qonly' if args.question_only else ''

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
        d_hat_path = EXP53_DIR / benchmark / f'd_hat_{args.model}.npz'
        if not d_hat_path.exists():
            print(f'  [SKIP] No d_hat at {d_hat_path}')
            continue
        d_hat_data = np.load(d_hat_path)
        if args.mem_predictor not in d_hat_data:
            print(f'  [SKIP] {args.mem_predictor} not found in {d_hat_path}')
            continue
        sim_pool.d_hat = d_hat_data[args.mem_predictor]

        # Load all cached c_hat files for this benchmark.
        c_hat_dir = EXP54_DIR / args.labels
        c_hat_all = {}
        bm_dir = c_hat_dir / benchmark
        if bm_dir.exists():
            for c_hat_path in sorted(bm_dir.glob(f'c_hat_*_{args.model}{qonly_suffix}.npz')):
                data = np.load(c_hat_path)
                for key in data:
                    if key not in c_hat_all:
                        c_hat_all[key] = data[key]

        available_corr = list(c_hat_all.keys())
        if not available_corr:
            print(f'  [SKIP] No c_hat files found for {benchmark}')
            print(f'  Looked in: {c_hat_dir / benchmark}')
            print(f'  Available keys: {list(c_hat_all.keys())}')
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
                    'n_replicates': args.n_replicates,
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
                        'n_replicates': args.n_replicates,
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

    # Keep output filenames tied to the run settings.
    bm_tag = args.benchmark if args.benchmark else 'all'
    labels_tag = args.labels.replace('_labels', '')
    suffix = f'{bm_tag}_{args.model}_{args.mem_predictor}_{labels_tag}_n{args.n}_g{args.gamma}_r{args.n_replicates}{qonly_suffix}'

    # Save combined results.
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(RESULTS_DIR / f'simulation_results_{suffix}.csv', index=False)
    results_df.to_parquet(RESULTS_DIR / f'simulation_results_{suffix}.parquet')

    # Write markdown tables.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    full_output = '\n\n'.join(tables)
    table_path = FIGURES_DIR / f'simulation_tables_{suffix}.md'
    table_path.write_text(full_output)
    print(f'\n{full_output}')
    print(f'\nSaved: {table_path}')


if __name__ == '__main__':
    main()
