"""Cross-benchmark correctness probe transfer simulation.

For each (source, target) benchmark pair:
  - fit correctness probes on the source calibration split
  - apply them to target simulation-split raw scores
  - run the target simulation with transferred c_hat

This measures how well correctness probes transfer across benchmarks.

Probes:
  - llama_platt: Platt scaling on Llama-3.1-8B confidence
  - roberta: Platt scaling on cached raw RoBERTa correctness scores from
    correctness/run_roberta.py

No GPU needed. Runs in minutes on CPU.

Usage:
  uv run python src/spiking/probe_transfer/run_corr_sim.py
  uv run python src/spiking/probe_transfer/run_corr_sim.py --source mmlu
  uv run python src/spiking/probe_transfer/run_corr_sim.py --regime both
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import (
    BENCHMARKS,
    BENCHMARK_LABELS,
    CORR_LABELS,
    DOSE_GROUPS,
    MODELS,
)
from hubble.corr_probes import LLMConfidenceProbe, RoBERTaCorrectnessProbe
from hubble.results import (
    load_cached_confidence,
    load_cached_roberta_scores,
    load_eval_item_pool,
)
from hubble.simulation import (
    DIFFICULTY_BINS,
    SamplerConfig,
    TestSet,
    combined_estimator,
    imputation_estimator,
    ipw_estimator,
    naive_estimator,
    sample_test_set,
    stratified_split,
)

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
PAPER_DIR = Path(__file__).resolve().parent.parent  # src/spiking/
DATA_RESULTS = PAPER_DIR / 'data_generation' / 'results'
EXP50_DIR = DATA_RESULTS
EXP53_DIR = PAPER_DIR / 'memorization' / 'results'
EXP13_DIR = DATA_RESULTS / 'confidence'
CORRECTNESS_CACHE_DIR = PAPER_DIR / 'correctness' / 'cache'

EXTERNAL_MODEL = 'meta-llama/Llama-3.1-8B'

CORR_PROBES = ['llama_platt', 'roberta']

ALL_BENCHMARKS = list(BENCHMARKS)


# Data loading

def load_item_pool(benchmark, model):
    """Load full ItemPool from release eval parquets."""
    return load_eval_item_pool(EXP50_DIR, benchmark, model)


def load_d_hat(benchmark, model, mem_probe):
    """Load cached d_hat for a benchmark."""
    path = EXP53_DIR / benchmark / f'd_hat_{model}.npz'
    data = np.load(path)
    assert mem_probe in data, (
        f'{mem_probe} not in {path} (available: {list(data.keys())})')
    return data[mem_probe]


def load_llm_confidence(benchmark):
    """Load LLM confidence aligned with eval parquet rows."""
    return load_cached_confidence(
        benchmark,
        EXP13_DIR,
        EXTERNAL_MODEL.split('/')[-1],
    )


def load_roberta_scores(benchmark, model):
    """Load cached raw RoBERTa correctness scores.

    Returns a 1D array aligned with eval parquet rows, or None if missing.
    """
    return load_cached_roberta_scores(
        CORRECTNESS_CACHE_DIR,
        benchmark,
        model,
        label_variant='standard',
    )


# Simulation

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


# Output formatting

def format_transfer_table(results_df, corr_probe, estimator, regime,
                          dose_group='high', difficulty_bin=None):
    """Format a source x target transfer matrix as a markdown table.

    Rows = source benchmark (c_hat source), Columns = target benchmark.
    Cells = RMSE (x100, in pp).
    """
    benchmarks = ALL_BENCHMARKS
    labels = [BENCHMARK_LABELS[b] for b in benchmarks]

    est_label = estimator.replace('_rmse', '').title()
    title = f'#### {est_label} ({CORR_LABELS.get(corr_probe, corr_probe)}) — {regime}'
    if regime == 'random':
        title += f' ({dose_group} dose)'
    else:
        title += f' ({difficulty_bin})'

    lines = [title, '']

    header = '| Source \\ Target | ' + ' | '.join(labels) + ' |'
    sep = '|' + '---|' * (len(labels) + 1)
    lines.extend([header, sep])

    for src_bm in benchmarks:
        row = f'| {BENCHMARK_LABELS[src_bm]} |'
        for tgt_bm in benchmarks:
            mask = (
                (results_df['source'] == src_bm)
                & (results_df['target'] == tgt_bm)
                & (results_df['corr_probe'] == corr_probe)
                & (results_df['regime'] == regime)
            )
            if regime == 'random':
                mask = mask & (results_df['dose_group'] == dose_group)
            else:
                mask = mask & (results_df['difficulty_bin'] == difficulty_bin)
            subset = results_df[mask]
            if len(subset) == 0:
                val = '—'
            else:
                val = f'{subset.iloc[0][estimator] * 100:.1f}'
                if src_bm == tgt_bm:
                    val = f'**{val}**'
            row += f' {val} |'
        lines.append(row)

    # Naive baseline row.
    row = '| Naive |'
    for tgt_bm in benchmarks:
        mask = (
            (results_df['target'] == tgt_bm)
            & (results_df['regime'] == regime)
        )
        if regime == 'random':
            mask = mask & (results_df['dose_group'] == dose_group)
        else:
            mask = mask & (results_df['difficulty_bin'] == difficulty_bin)
        subset = results_df[mask]
        if len(subset) == 0:
            val = '—'
        else:
            val = f'{subset.iloc[0]["naive_rmse"] * 100:.1f}'
        row += f' {val} |'
    lines.append(row)

    return '\n'.join(lines)


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=500, help='Test set size')
    parser.add_argument('--gamma', type=float, default=0.3, help='Contamination rate')
    parser.add_argument('--n-replicates', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model', type=str, default='8b-500b', choices=MODELS)
    parser.add_argument('--source', type=str, nargs='+', default=None,
                        choices=ALL_BENCHMARKS,
                        help='Source benchmark(s) for c_hat (default: all)')
    parser.add_argument('--target', type=str, nargs='+', default=None,
                        choices=ALL_BENCHMARKS,
                        help='Target benchmark(s) (default: all)')
    parser.add_argument('--mem-probe', type=str, default='min_k_plus_plus',
                        help='Memorization probe for d_hat (default: min_k_plus_plus)')
    parser.add_argument('--regime', type=str, default='random',
                        choices=['random', 'correlated', 'both'],
                        help='Contamination regime (default: random)')
    args = parser.parse_args()

    source_benchmarks = args.source if args.source else ALL_BENCHMARKS
    target_benchmarks = args.target if args.target else ALL_BENCHMARKS
    all_needed = set(source_benchmarks) | set(target_benchmarks)

    # Pre-load item pools and raw confidence scores.
    print('Loading item pools and raw confidence scores...')
    benchmark_data = {}
    for bm in all_needed:
        pool = load_item_pool(bm, args.model)
        cal_pool, sim_pool, cal_idx, sim_idx = stratified_split(pool)
        clean_cal_mask = cal_pool.duplicates == 0
        labels_cal_clean = cal_pool.y_clean[clean_cal_mask]

        benchmark_data[bm] = {
            'cal_idx': cal_idx,
            'sim_idx': sim_idx,
            'sim_pool': sim_pool,
            'clean_cal_mask': clean_cal_mask,
            'labels_cal_clean': labels_cal_clean,
        }

        llm_conf = load_llm_confidence(bm)
        if llm_conf is not None:
            assert len(llm_conf) == pool.n_items, (
                f'{bm}: LLM confidence length {len(llm_conf)} != pool size {pool.n_items}')
            benchmark_data[bm]['llm_conf'] = llm_conf

        roberta_scores = load_roberta_scores(bm, args.model)
        if roberta_scores is not None:
            assert len(roberta_scores) == pool.n_items, (
                f'{bm}: RoBERTa score length {len(roberta_scores)} != pool size {pool.n_items}')
            benchmark_data[bm]['roberta_scores'] = roberta_scores

        avail = [
            p for p in CORR_PROBES
            if (p == 'llama_platt' and 'llm_conf' in benchmark_data[bm])
            or (p == 'roberta' and 'roberta_scores' in benchmark_data[bm])
        ]
        print(f'  {bm}: {pool.n_items} items '
              f'(cal={len(cal_idx)}, sim={len(sim_idx)}, probes={avail})')

    # Load d_hat for target benchmarks.
    print(f'\nLoading d_hat ({args.mem_probe})...')
    for tgt in target_benchmarks:
        if tgt not in benchmark_data:
            continue
        sim_pool = benchmark_data[tgt]['sim_pool']
        d_hat = load_d_hat(tgt, args.model, args.mem_probe)
        assert len(d_hat) == sim_pool.n_items, (
            f'{tgt}: d_hat length {len(d_hat)} != sim pool size {sim_pool.n_items}')
        benchmark_data[tgt]['d_hat'] = d_hat
        print(f'  {tgt}: {sim_pool.n_items} items')

    all_rows = []

    for src in source_benchmarks:
        if src not in benchmark_data:
            continue
        src_data = benchmark_data[src]

        for tgt in target_benchmarks:
            if tgt not in benchmark_data or 'd_hat' not in benchmark_data[tgt]:
                continue
            tgt_data = benchmark_data[tgt]
            sim_pool = tgt_data['sim_pool']
            sim_pool.d_hat = tgt_data['d_hat']

            # Fit on the source cal split; predict on the target sim split.
            c_hat_dict = {}

            # Llama Platt on LLM confidence scores.
            if 'llm_conf' in src_data and 'llm_conf' in tgt_data:
                probe = LLMConfidenceProbe(EXTERNAL_MODEL)
                src_conf_cal = src_data['llm_conf'][src_data['cal_idx']][src_data['clean_cal_mask']]
                probe.fit(src_conf_cal, src_data['labels_cal_clean'])
                tgt_conf_sim = tgt_data['llm_conf'][tgt_data['sim_idx']]
                c_hat_dict['llama_platt'] = probe.predict_proba(tgt_conf_sim)[:, 1]

            if 'roberta_scores' in src_data and 'roberta_scores' in tgt_data:
                probe = RoBERTaCorrectnessProbe(
                    model_dir=CORRECTNESS_CACHE_DIR / '_transfer_platt',
                )
                src_scores_cal = src_data['roberta_scores'][src_data['cal_idx']][src_data['clean_cal_mask']]
                probe.fit(src_scores_cal, src_data['labels_cal_clean'])
                tgt_scores_sim = tgt_data['roberta_scores'][tgt_data['sim_idx']]
                c_hat_dict['roberta'] = probe.predict_proba(tgt_scores_sim)[:, 1]

            available_corr = [c for c in CORR_PROBES if c in c_hat_dict]
            if not available_corr:
                continue

            print(f'\n  {src} -> {tgt}: {available_corr}')

            # Random regime.
            if args.regime in ('random', 'both'):
                for dose_group in DOSE_GROUPS:
                    for corr_probe in available_corr:
                        sim_pool.c_hat = c_hat_dict[corr_probe]
                        result = run_sim(
                            sim_pool, 'random', dose_group, args.n, args.gamma,
                            args.n_replicates, args.seed,
                        )
                        all_rows.append({
                            'source': src, 'target': tgt,
                            'model': args.model,
                            'regime': 'random', 'dose_group': dose_group,
                            'difficulty_bin': None,
                            'mem_probe': args.mem_probe,
                            'corr_probe': corr_probe,
                            **result,
                        })
                    print(f'    random/{dose_group}: done')

            # Correlated regime.
            has_confidence = sim_pool.confidence is not None
            if args.regime in ('correlated', 'both') and has_confidence:
                for difficulty_bin in DIFFICULTY_BINS:
                    for corr_probe in available_corr:
                        sim_pool.c_hat = c_hat_dict[corr_probe]
                        result = run_sim(
                            sim_pool, 'correlated', 'high', args.n, args.gamma,
                            args.n_replicates, args.seed,
                            difficulty_bin=difficulty_bin,
                        )
                        all_rows.append({
                            'source': src, 'target': tgt,
                            'model': args.model,
                            'regime': 'correlated', 'dose_group': 'high',
                            'difficulty_bin': difficulty_bin,
                            'mem_probe': args.mem_probe,
                            'corr_probe': corr_probe,
                            **result,
                        })
                    print(f'    correlated/{difficulty_bin}: done')

    # Save raw results.
    results_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'corr_transfer_simulation_results.csv', index=False)
    results_df.to_parquet(RESULTS_DIR / 'corr_transfer_simulation_results.parquet')
    print(f'\nSaved {len(results_df)} rows to {RESULTS_DIR}/corr_transfer_simulation_results.*')

    # Generate transfer matrix tables.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tables = []

    for corr_probe in CORR_PROBES:
        if corr_probe not in results_df['corr_probe'].values:
            continue
        for estimator in ['imputation_rmse', 'combined_rmse', 'ipw_rmse']:
            if args.regime in ('random', 'both'):
                for dose_group in DOSE_GROUPS:
                    tables.append(format_transfer_table(
                        results_df, corr_probe, estimator, 'random',
                        dose_group=dose_group))
            if args.regime in ('correlated', 'both'):
                for difficulty_bin in DIFFICULTY_BINS:
                    tables.append(format_transfer_table(
                        results_df, corr_probe, estimator, 'correlated',
                        difficulty_bin=difficulty_bin))

    full_output = '\n\n'.join(tables)
    table_path = FIGURES_DIR / 'corr_transfer_tables.md'
    table_path.write_text(full_output)
    print(f'\n{full_output}')
    print(f'\nSaved tables to {table_path}')


if __name__ == '__main__':
    main()
