"""Cross-benchmark memorization probe transfer simulation.

For each (source, target) benchmark pair:
  - fit MIA probes on the source calibration split
  - apply them to target simulation-split raw scores
  - run IPW simulation on the target pool with transferred d_hat

This measures how well memorization probes transfer across benchmarks.

MIA probes: Platt scaling fitted on source cal scores, applied to target sim scores.
Hidden-state probes: logistic regression fitted on source cal features, applied
to target sim features.

No GPU needed. Runs in minutes on CPU.

Usage:
  uv run python src/spiking/probe_transfer/run_mem_sim.py
  uv run python src/spiking/probe_transfer/run_mem_sim.py --source mmlu
  uv run python src/spiking/probe_transfer/run_mem_sim.py --source wikipedia
  uv run python src/spiking/probe_transfer/run_mem_sim.py --corr-probe roberta
  uv run python src/spiking/probe_transfer/run_mem_sim.py --labels perturbed
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import (
    ATTACKS,
    BENCHMARK_EXP11_MAP as EXP11_MAP,
    BENCHMARK_LABELS,
    DOSE_GROUPS,
    MEM_LABELS,
    MEM_PROBES,
    MODELS,
)
from hubble.mem_probes import MIAProbe
from hubble.probes import PROBES
from hubble.simulation import (
    DIFFICULTY_BINS,
    ItemPool,
    SamplerConfig,
    TestSet,
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
EXP11_SCORES = DATA_RESULTS / 'all_scores.parquet'
EXP12_FEATURES = DATA_RESULTS / 'features'
EXP54_DIR = PAPER_DIR / 'correctness' / 'results'

CORR_PROBES = {
    'llama_platt': {
        'file_pattern': 'c_hat_llama_{model}.npz',
        'key': 'platt',
        'fallback_key': 'llm_platt',
    },
    'roberta': {
        'file_pattern': 'c_hat_roberta_{model}.npz',
        'key': 'roberta',
        'fallback_key': 'roberta_platt',
    },
}

# Wikipedia uses continuous loss and has no c_hat.
LOSS_BENCHMARKS = {'wikipedia'}

ALL_BENCHMARKS = list(BENCHMARK_LABELS.keys())


# Data loading

def load_item_pool(benchmark, model):
    """Load the full ItemPool from release eval parquets.

    Loss-based benchmarks only need duplicate labels for probe fitting, so they
    get dummy outcomes here.
    """
    tag = f'hubble-{model}_toks'
    std_path = EXP50_DIR / benchmark / f'eval_{tag}-standard-hf.parquet'
    prt_path = EXP50_DIR / benchmark / f'eval_{tag}-perturbed-hf.parquet'

    std_df = pd.read_parquet(std_path)

    if benchmark in LOSS_BENCHMARKS:
        n = len(std_df)
        duplicates = std_df['duplicates'].values.astype(int)
        return ItemPool(
            y_observed=np.zeros(n, dtype=int),
            y_clean=np.zeros(n, dtype=int),
            duplicates=duplicates,
        )

    acc_prefix = 'acc' if f'acc_{tag}-standard-hf' in std_df.columns else 'exact_match'
    confidence_col = f'confidence_{tag}-standard-hf'
    has_confidence = confidence_col in std_df.columns

    return ItemPool.from_eval_parquets(
        str(std_path), str(prt_path),
        acc_clean_col=f'{acc_prefix}_{tag}-standard-hf',
        acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
        confidence_col=confidence_col if has_confidence else None,
    )


def load_target_sim_pool(benchmark, model):
    """Load sim pool for a target benchmark.

    For loss-based benchmarks (Wikipedia), uses continuous loss values as
    y_observed/y_clean instead of binary accuracy. The IPW and naive estimators
    work identically on continuous outcomes.
    """
    tag = f'hubble-{model}_toks'
    std_path = EXP50_DIR / benchmark / f'eval_{tag}-standard-hf.parquet'
    prt_path = EXP50_DIR / benchmark / f'eval_{tag}-perturbed-hf.parquet'

    std_df = pd.read_parquet(std_path)
    prt_df = pd.read_parquet(prt_path)

    if benchmark in LOSS_BENCHMARKS:
        loss_std_col = f'loss_{tag}-standard-hf'
        loss_prt_col = f'loss_{tag}-perturbed-hf'
        duplicates = std_df['duplicates'].values.astype(int)
        y_clean = std_df[loss_std_col].values
        y_observed = np.where(
            duplicates > 0, prt_df[loss_prt_col].values, y_clean)
        full_pool = ItemPool(
            y_observed=y_observed, y_clean=y_clean, duplicates=duplicates,
        )
    else:
        acc_prefix = 'acc' if f'acc_{tag}-standard-hf' in std_df.columns else 'exact_match'
        confidence_col = f'confidence_{tag}-standard-hf'
        has_confidence = confidence_col in std_df.columns

        full_pool = ItemPool.from_eval_parquets(
            str(std_path), str(prt_path),
            acc_clean_col=f'{acc_prefix}_{tag}-standard-hf',
            acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
            confidence_col=confidence_col if has_confidence else None,
        )

    _, sim_pool, _, _ = stratified_split(full_pool)
    return sim_pool


def load_mia_scores(benchmark, model):
    """Load cached raw MIA attack scores for a benchmark.

    Returns (scores_matrix, attack_names) where scores_matrix is (n_items, n_attacks).
    """
    mapping = EXP11_MAP[benchmark]
    all_scores = pd.read_parquet(EXP11_SCORES)
    mask = ((all_scores['benchmark'] == mapping['exp11_benchmark'])
            & (all_scores['model'] == model))
    if mapping['exp11_format'] is not None:
        mask = mask & (all_scores['format'] == mapping['exp11_format'])
    df = all_scores[mask].reset_index(drop=True)
    return df[ATTACKS].values, ATTACKS


def load_hidden_features(benchmark, model, pool_name):
    """Load cached hidden-state features.

    Returns features array of shape (n_items, hidden_dim).
    """
    mapping = EXP11_MAP[benchmark]
    exp12_benchmark = mapping['exp11_benchmark']
    exp12_format = mapping['exp11_format']
    label = f'hubble-{model}_toks-perturbed-hf'
    feat_path = EXP12_FEATURES / exp12_benchmark / f'features_{label}_{pool_name}.npz'
    meta_path = EXP12_FEATURES / exp12_benchmark / f'meta_{label}.parquet'

    features = np.load(feat_path)['hidden_states']
    if exp12_format is not None:
        meta = pd.read_parquet(meta_path)
        mask = (meta['format'] == exp12_format).values
        features = features[mask]
    return features


def load_c_hat(benchmark, model, labels_dir, corr_probe):
    """Load cached c_hat for a benchmark.

    Returns c_hat array (sim split) or None if not available.
    """
    spec = CORR_PROBES[corr_probe]
    filename = spec['file_pattern'].format(model=model)
    path = EXP54_DIR / labels_dir / benchmark / filename
    if not path.exists():
        return None
    data = np.load(path)
    key = spec['key']
    fallback = spec.get('fallback_key')
    if key in data:
        return data[key]
    if fallback and fallback in data:
        return data[fallback]
    return None


# Simulation

def run_sim(pool, regime, dose_group, n, gamma, n_replicates, seed,
            difficulty_bin='hard'):
    """Run Monte Carlo simulation, return RMSE for naive and IPW."""
    cfg = SamplerConfig(
        regime=regime, n=n, gamma=gamma,
        dose_group=dose_group, difficulty_bin=difficulty_bin,
    )

    rng = np.random.default_rng(seed)
    naive_errors = []
    ipw_errors = []

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
        naive_errors.append((naive_estimator(ts) - gt) ** 2)
        ipw_errors.append((ipw_estimator(ts) - gt) ** 2)

    return {
        'naive_rmse': np.sqrt(np.mean(naive_errors)),
        'ipw_rmse': np.sqrt(np.mean(ipw_errors)),
    }


# Output formatting

def format_transfer_table(results_df, mem_probe, regime,
                          dose_group='high', difficulty_bin=None):
    """Format a source x target transfer matrix as a markdown table.

    Rows = source benchmark (d_hat source), Columns = target benchmark.
    Cells = IPW RMSE (x100, in pp).
    """
    benchmarks = ALL_BENCHMARKS
    labels = [BENCHMARK_LABELS[b] for b in benchmarks]

    title = f'#### IPW ({MEM_LABELS.get(mem_probe, mem_probe)}) — {regime}'
    if regime == 'random':
        title += f' ({dose_group} dose)'
    else:
        title += f' ({difficulty_bin})'

    lines = [title, '']

    header = '| Source \\\\ Target | ' + ' | '.join(labels) + ' |'
    sep = '|' + '---|' * (len(labels) + 1)
    lines.extend([header, sep])

    for src_bm in benchmarks:
        row = f'| {BENCHMARK_LABELS[src_bm]} |'
        for tgt_bm in benchmarks:
            mask = (
                (results_df['source'] == src_bm)
                & (results_df['target'] == tgt_bm)
                & (results_df['mem_probe'] == mem_probe)
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
                val = f'{subset.iloc[0]["ipw_rmse"] * 100:.1f}'
                if src_bm == tgt_bm:
                    val = f'**{val}**'
            row += f' {val} |'
        lines.append(row)

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


def format_average_transfer_table(results_df, mem_probe):
    """Format a source x target transfer matrix averaged across all settings.

    Averages IPW RMSE across:
      - random: low, mid, high
      - correlated: easy, medium, hard

    Rows = source benchmark (d_hat source), Columns = target benchmark.
    Cells = mean IPW RMSE (x100, in pp).
    """
    benchmarks = ALL_BENCHMARKS
    labels = [BENCHMARK_LABELS[b] for b in benchmarks]

    title = f'#### IPW ({MEM_LABELS.get(mem_probe, mem_probe)}) — average across all settings'
    lines = [title, '']

    header = '| Source \\\\ Target | ' + ' | '.join(labels) + ' |'
    sep = '|' + '---|' * (len(labels) + 1)
    lines.extend([header, sep])

    for src_bm in benchmarks:
        row = f'| {BENCHMARK_LABELS[src_bm]} |'
        for tgt_bm in benchmarks:
            subset = results_df[
                (results_df['source'] == src_bm)
                & (results_df['target'] == tgt_bm)
                & (results_df['mem_probe'] == mem_probe)
            ]

            if len(subset) == 0:
                val = '—'
            else:
                val = f'{subset["ipw_rmse"].mean() * 100:.1f}'
                if src_bm == tgt_bm:
                    val = f'**{val}**'
            row += f' {val} |'
        lines.append(row)

    row = '| Naive |'
    for tgt_bm in benchmarks:
        subset = results_df[results_df['target'] == tgt_bm]
        if len(subset) == 0:
            val = '—'
        else:
            naive_vals = subset[
                ['regime', 'dose_group', 'difficulty_bin', 'target', 'naive_rmse']
            ].drop_duplicates()['naive_rmse']
            val = f'{naive_vals.mean() * 100:.1f}'
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
    parser.add_argument('--model', type=str, default='8b-500b')
    parser.add_argument('--source', type=str, default=None,
                        choices=ALL_BENCHMARKS,
                        help='Single source benchmark for d_hat (default: all)')
    parser.add_argument('--target', type=str, default=None,
                        choices=ALL_BENCHMARKS,
                        help='Single target benchmark (default: all)')
    parser.add_argument('--corr-probe', type=str, default='llama_platt',
                        choices=list(CORR_PROBES.keys()),
                        help='Correctness probe for c_hat (default: llama_platt)')
    parser.add_argument('--labels', type=str, default='standard_labels',
                        choices=['standard_labels', 'perturbed'],
                        help='c_hat label directory (default: standard_labels)')
    parser.add_argument('--regime', type=str, default='random',
                        choices=['random', 'correlated', 'both'],
                        help='Contamination regime (default: random)')
    args = parser.parse_args()

    source_benchmarks = [args.source] if args.source else ALL_BENCHMARKS
    target_benchmarks = [args.target] if args.target else ALL_BENCHMARKS

    print('Loading raw MIA scores and item pools...')
    benchmark_data = {}
    all_needed = set(source_benchmarks) | set(target_benchmarks)
    for bm in all_needed:
        pool = load_item_pool(bm, args.model)
        cal_pool, sim_pool, cal_idx, sim_idx = stratified_split(pool)
        labels_cal = (cal_pool.duplicates > 0).astype(int)
        scores_matrix, _ = load_mia_scores(bm, args.model)
        assert len(scores_matrix) == pool.n_items, (
            f'{bm}: {len(scores_matrix)} scores vs {pool.n_items} items')
        benchmark_data[bm] = {
            'scores': scores_matrix,
            'cal_idx': cal_idx,
            'sim_idx': sim_idx,
            'labels_cal': labels_cal,
        }
        print(f'  {bm}: {pool.n_items} items '
              f'(cal={len(cal_idx)}, sim={len(sim_idx)})')

    print(f'\nLoading target pools and c_hat ({args.labels}, {args.corr_probe})...')
    target_pools = {}
    target_c_hats = {}
    for tgt in target_benchmarks:
        sim_pool = load_target_sim_pool(tgt, args.model)
        if tgt in LOSS_BENCHMARKS:
            c_hat = np.zeros(sim_pool.n_items)
            print(f'  {tgt}: {sim_pool.n_items} items (loss-based, dummy c_hat)')
        else:
            c_hat = load_c_hat(tgt, args.model, args.labels, args.corr_probe)
            if c_hat is None:
                print(f'  [SKIP] {tgt}: no {args.corr_probe} c_hat found')
                continue
            assert len(c_hat) == sim_pool.n_items, (
                f'{tgt}: c_hat length {len(c_hat)} != sim pool size {sim_pool.n_items}')
            print(f'  {tgt}: {sim_pool.n_items} items')
        target_pools[tgt] = sim_pool
        target_c_hats[tgt] = c_hat

    print('\nLoading hidden-state features...')
    benchmark_features = {}
    for bm in all_needed:
        benchmark_features[bm] = {}
        for probe_name, probe_cls in PROBES.items():
            pool_name = probe_cls.pool.__name__
            try:
                features = load_hidden_features(bm, args.model, pool_name)
                benchmark_features[bm][probe_name] = features
                print(f'  {bm}/{probe_name}: {features.shape}')
            except FileNotFoundError:
                print(f'  {bm}/{probe_name}: SKIPPED (no cached features)')

    all_rows = []

    for src in source_benchmarks:
        src_data = benchmark_data[src]

        for tgt in target_benchmarks:
            if tgt not in target_pools:
                continue

            tgt_data = benchmark_data[tgt]
            sim_pool = target_pools[tgt]
            sim_pool.c_hat = target_c_hats[tgt]

            d_hat_dict = {}
            for i, attack in enumerate(ATTACKS):
                probe = MIAProbe(attack)
                src_scores = src_data['scores'][:, i]
                probe.fit(src_scores[src_data['cal_idx']], src_data['labels_cal'])
                tgt_scores = tgt_data['scores'][:, i]
                d_hat_dict[attack] = probe.predict_proba(
                    tgt_scores[tgt_data['sim_idx']])

            for probe_name, probe_cls in PROBES.items():
                if probe_name not in benchmark_features.get(src, {}):
                    continue
                if probe_name not in benchmark_features.get(tgt, {}):
                    continue
                src_feats = benchmark_features[src][probe_name]
                tgt_feats = benchmark_features[tgt][probe_name]
                probe = probe_cls()
                probe.fit(src_feats[src_data['cal_idx']], src_data['labels_cal'])
                d_hat_dict[probe_name] = probe.predict_proba(
                    tgt_feats[tgt_data['sim_idx']])

            available_probes = [p for p in MEM_PROBES if p in d_hat_dict]
            print(f'\n  {src} -> {tgt}: {len(available_probes)} probes')

            if args.regime in ('random', 'both'):
                for dose_group in DOSE_GROUPS:
                    for mem_probe in available_probes:
                        sim_pool.d_hat = d_hat_dict[mem_probe]
                        result = run_sim(
                            sim_pool, 'random', dose_group, args.n, args.gamma,
                            args.n_replicates, args.seed,
                        )
                        all_rows.append({
                            'source': src, 'target': tgt,
                            'model': args.model,
                            'regime': 'random', 'dose_group': dose_group,
                            'difficulty_bin': None,
                            'mem_probe': mem_probe,
                            'corr_probe': args.corr_probe,
                            **result,
                        })
                    print(f'    random/{dose_group}: done')

            has_confidence = sim_pool.confidence is not None
            if args.regime in ('correlated', 'both') and has_confidence:
                for difficulty_bin in DIFFICULTY_BINS:
                    for mem_probe in available_probes:
                        sim_pool.d_hat = d_hat_dict[mem_probe]
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
                            'mem_probe': mem_probe,
                            'corr_probe': args.corr_probe,
                            **result,
                        })
                    print(f'    correlated/{difficulty_bin}: done')

    results_df = pd.DataFrame(all_rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_DIR / 'transfer_simulation_results.csv', index=False)
    results_df.to_parquet(RESULTS_DIR / 'transfer_simulation_results.parquet')
    print(f'\nSaved {len(results_df)} rows to {RESULTS_DIR}/transfer_simulation_results.*')

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    tables = []

    for mem_probe in MEM_PROBES:
        if mem_probe not in results_df['mem_probe'].values:
            continue
        if args.regime in ('random', 'both'):
            for dose_group in DOSE_GROUPS:
                tables.append(format_transfer_table(
                    results_df, mem_probe, 'random', dose_group=dose_group))
        if args.regime in ('correlated', 'both'):
            for difficulty_bin in DIFFICULTY_BINS:
                tables.append(format_transfer_table(
                    results_df, mem_probe, 'correlated',
                    difficulty_bin=difficulty_bin))

        tables.append(format_average_transfer_table(results_df, mem_probe))

    full_output = '\n\n'.join(tables)
    table_path = FIGURES_DIR / 'transfer_tables.md'
    table_path.write_text(full_output)
    print(f'\n{full_output}')
    print(f'\nSaved tables to {table_path}')


if __name__ == '__main__':
    main()
