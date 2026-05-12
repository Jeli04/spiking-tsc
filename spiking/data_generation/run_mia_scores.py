"""Compute MIA attack scores across all models and benchmarks.

Two-phase workflow:
  Phase 1 (score): GPU. Compute attack scores, cache to parquet.
  Phase 2 (combine): CPU. Merge per-(attack, benchmark, model) scores into one wide file.

Usage:
  # Score one attack/model pair across all benchmarks.
  uv run python src/spiking/data_generation/run_mia_scores.py score --attack loss --model 0

  # Score one benchmark.
  uv run python src/spiking/data_generation/run_mia_scores.py score --attack loss --model 0 --benchmark wikipedia

  # SLURM array: 5 attacks x 4 models = 20 tasks.
  sbatch --array=0-19 slurm/run_gpu.sbatch src/spiking/data_generation/run_mia_scores.py score

  # Combine all cached scores.
  uv run python src/spiking/data_generation/run_mia_scores.py combine

  # Combine one benchmark and merge into the existing table.
  uv run python src/spiking/data_generation/run_mia_scores.py combine --benchmark wikipedia
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from hubble.data import BENCHMARK_LOADERS
from hubble.runner import HUBBLE_MODELS, _model_label
from mia.attacks import ATTACKS

# Constants

RESULTS_DIR = Path(__file__).parent / 'results'
SCORES_DIR = RESULTS_DIR / 'scores'

ATTACK_NAMES = ['loss', 'zlib', 'min_k', 'min_k_plus_plus', 'reference']
PERTURBED_MODELS = [m for m in HUBBLE_MODELS if 'perturbed' in m]
BENCHMARKS = ['winogrande', 'mmlu', 'piqa', 'popqa', 'hellaswag', 'wikipedia']
MODEL_ORDER = ['1b-100b', '1b-500b', '8b-100b', '8b-500b']

# Phase 1: Score (GPU)


def phase_score(attack_name: str, model_idx: int, benchmarks=None):
    """Compute MIA scores for one attack x one model across benchmarks."""
    from hubble.eval import load_model  # deferred import (needs GPU)

    if benchmarks is None:
        benchmarks = BENCHMARKS

    model_id = PERTURBED_MODELS[model_idx]
    label = _model_label(model_id)
    attack_fn = ATTACKS[attack_name]

    print(f'Scoring: attack={attack_name}, model={label}')

    # Load the target model, plus a reference model when needed.
    attack_kwargs = {}
    if attack_name == 'reference':
        ref_id = model_id.replace('perturbed', 'standard')
        print(f'  Reference model: {_model_label(ref_id)}')
        model, tokenizer = load_model(model_id)
        ref_model, ref_tokenizer = load_model(ref_id)
        attack_kwargs = {'ref_model': ref_model, 'ref_tokenizer': ref_tokenizer}
    else:
        model, tokenizer = load_model(model_id)

    for benchmark in benchmarks:
        out_dir = SCORES_DIR / attack_name / benchmark
        out_dir.mkdir(parents=True, exist_ok=True)
        cache_path = out_dir / f'scores_{label}.parquet'

        if cache_path.exists():
            print(f'  Cached: {cache_path}')
            continue

        df = BENCHMARK_LOADERS[benchmark]()

        scores = []
        for text in tqdm(df['text'], desc=f'  {benchmark}'):
            scores.append(attack_fn(model, tokenizer, text, **attack_kwargs))

        out_df = pd.DataFrame({
            'orig_idx': df['orig_idx'].values if 'orig_idx' in df.columns else range(len(df)),
            'duplicates': df['duplicates'].values,
            'split': df['split'].values,
            'score': scores,
        })
        if 'format' in df.columns:
            out_df['format'] = df['format'].values

        out_df.to_parquet(cache_path)
        print(f'  Saved: {cache_path} ({len(out_df)} rows)')


# Phase 2: Combine (CPU)


def _parse_score_label(filename: str) -> str:
    """'scores_hubble-1b-100b_toks-perturbed-hf.parquet' -> '1b-100b'."""
    m = re.match(r'scores_hubble-(\d+b)-(\d+b)_toks', filename)
    return f'{m.group(1)}-{m.group(2)}' if m else filename


def phase_combine(benchmarks=None, models=None):
    """Merge per-(attack, benchmark, model) score parquets into one wide file.

    If benchmarks is specified, only process those benchmarks and merge
    into the existing all_scores.parquet (if it exists).
    If models is specified, only require/process those models.
    """
    if benchmarks is None:
        benchmarks = BENCHMARKS
    if models is None:
        models = MODEL_ORDER

    # Load score parquets for the requested benchmarks.
    frames = []
    found = set()
    for parquet in sorted(SCORES_DIR.glob('*/*/*.parquet')):
        attack = parquet.parent.parent.name
        benchmark = parquet.parent.name
        if benchmark not in benchmarks:
            continue
        model = _parse_score_label(parquet.name)
        if model not in models:
            continue
        found.add((attack, benchmark, model))

        df = pd.read_parquet(parquet)
        df = df.rename(columns={'score': attack})
        df['benchmark'] = benchmark
        df['model'] = model
        frames.append(df)

    # Check completeness for the requested benchmarks.
    expected = {
        (a, b, m)
        for a in ATTACK_NAMES
        for b in benchmarks
        for m in models
    }
    missing = expected - found
    if missing:
        print(f'ERROR: {len(missing)} missing score files:')
        for a, b, m in sorted(missing):
            print(f'  {a}/{b}/{m}')
        sys.exit(1)
    extra = found - expected
    if extra:
        print(f'WARNING: {len(extra)} unexpected score files:')
        for a, b, m in sorted(extra):
            print(f'  {a}/{b}/{m}')
    print(f'Found {len(found)} score files '
          f'({len(ATTACK_NAMES)} attacks x {len(benchmarks)} '
          f'benchmarks x {len(models)} models)')

    # Merge attacks per benchmark/model pair.
    groups = {}
    for df in frames:
        key = (df['benchmark'].iloc[0], df['model'].iloc[0])
        if key not in groups:
            groups[key] = df
        else:
            attack_col = [c for c in df.columns
                          if c not in groups[key].columns][0]
            groups[key][attack_col] = df[attack_col].values

    new_data = pd.concat(groups.values(), ignore_index=True)

    # Reorder columns.
    meta_cols = ['benchmark', 'model', 'orig_idx', 'duplicates', 'split']
    if 'format' in new_data.columns:
        meta_cols.append('format')
    attack_cols = sorted(c for c in new_data.columns if c not in meta_cols)
    new_data = new_data[meta_cols + attack_cols]

    # Sanity checks on new data.
    n_nulls = new_data[attack_cols].isnull().sum()
    if n_nulls.any():
        print(f'ERROR: null scores found:\n{n_nulls[n_nulls > 0]}')
        sys.exit(1)

    assert set(attack_cols) == set(ATTACK_NAMES), (
        f'Attack columns mismatch: {attack_cols} vs {ATTACK_NAMES}')

    EXPECTED_ROWS = {
        'winogrande': 16002,  # infill and MCQ formats
        'wikipedia': 2884,
    }
    for (bm, mdl), grp in new_data.groupby(['benchmark', 'model']):
        n = len(grp)
        expected_n = EXPECTED_ROWS.get(bm, 8001)
        assert n == expected_n, (
            f'{bm}/{mdl}: expected {expected_n} rows, got {n}')

    print(f'Sanity checks passed: {len(new_data)} new rows, '
          f'no nulls, {len(attack_cols)} attack columns')

    # Merge with existing all_scores when this is a partial combine.
    out_path = RESULTS_DIR / 'all_scores.parquet'
    is_partial = (set(benchmarks) != set(BENCHMARKS)
                  or set(models) != set(MODEL_ORDER))
    if is_partial and out_path.exists():
        existing = pd.read_parquet(out_path)
        # Replace rows for the benchmarks/models just recomputed.
        existing = existing[~(existing['benchmark'].isin(benchmarks)
                              & existing['model'].isin(models))]
        combined = pd.concat([existing, new_data], ignore_index=True)
        print(f'Merged with existing: {len(existing)} old rows + '
              f'{len(new_data)} new rows = {len(combined)} total')
    else:
        combined = new_data

    # Save.
    combined.to_parquet(out_path)
    print(f'Saved to {out_path}')


# CLI


def main():
    parser = argparse.ArgumentParser(
        description='MIA scores across all models and benchmarks')
    sub = parser.add_subparsers(dest='command')

    sp = sub.add_parser('score', help='Compute MIA scores (GPU)')
    sp.add_argument('--attack', choices=ATTACK_NAMES,
                    help='Attack to run. Inferred from SLURM_ARRAY_TASK_ID if omitted.')
    sp.add_argument('--model', type=int, choices=range(len(PERTURBED_MODELS)),
                    help='Perturbed model index (0-3). Inferred from SLURM_ARRAY_TASK_ID if omitted.')
    sp.add_argument('--benchmark', choices=BENCHMARKS, nargs='+', default=None,
                    help='Benchmark(s) to score (default: all)')

    cp = sub.add_parser('combine', help='Combine cached scores into one file (CPU)')
    cp.add_argument('--benchmark', choices=BENCHMARKS, nargs='+', default=None,
                    help='Benchmark(s) to combine. Merges into existing all_scores if subset.')
    cp.add_argument('--model-filter', choices=MODEL_ORDER, nargs='+', default=None,
                    help='Model(s) to combine (default: all). Merges into existing all_scores if subset.')

    args = parser.parse_args()

    if args.command == 'score':
        attack = args.attack
        model_idx = args.model

        # SLURM_ARRAY_TASK_ID maps to attack_idx * 4 + model_idx.
        if attack is None or model_idx is None:
            task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
            if task_id is not None:
                tid = int(task_id)
                attack = ATTACK_NAMES[tid // len(PERTURBED_MODELS)]
                model_idx = tid % len(PERTURBED_MODELS)
            else:
                parser.error(
                    'Provide --attack and --model, or set SLURM_ARRAY_TASK_ID')

        phase_score(attack, model_idx, benchmarks=args.benchmark)

    elif args.command == 'combine':
        phase_combine(benchmarks=args.benchmark, models=args.model_filter)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
