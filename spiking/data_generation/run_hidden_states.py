"""Extract hidden-state features across all models and benchmarks.

Two-phase workflow:
  Phase 1 (extract): GPU. Extract pooled hidden states, cache to .npz.
  Phase 2 (verify):  CPU. Check all expected caches exist.

Usage:
  # Extract features for one model across all benchmarks.
  uv run python spiking/data_generation/run_hidden_states.py extract --model 0

  # Extract features for one model by compact task id.
  HUBBLE_TASK_ID=0 uv run python spiking/data_generation/run_hidden_states.py extract

  # Verify all caches.
  uv run python spiking/data_generation/run_hidden_states.py verify
"""

import argparse
import os
import sys
import time
from pathlib import Path

from hubble.data import BENCHMARK_LOADERS
from hubble.predictors import PREDICTORS
from hubble.runner import HUBBLE_MODELS, _model_label

# Constants

RESULTS_DIR = Path(__file__).parent / 'results'
FEATURES_DIR = RESULTS_DIR / 'features'

PERTURBED_MODELS = [m for m in HUBBLE_MODELS if 'perturbed' in m]
BENCHMARKS = ['winogrande', 'mmlu', 'piqa', 'popqa', 'hellaswag', 'wikipedia']

def _feature_path(benchmark: str, label: str, pool_name: str) -> Path:
    return FEATURES_DIR / benchmark / f'features_{label}_{pool_name}.npz'


def _meta_path(benchmark: str, label: str) -> Path:
    return FEATURES_DIR / benchmark / f'meta_{label}.parquet'


# Phase 1: Extract (GPU)


def phase_extract(model_idx: int):
    """Extract hidden-state features for one model across all benchmarks."""
    from hubble.eval import load_model  # deferred import (needs GPU)

    model_id = PERTURBED_MODELS[model_idx]
    label = _model_label(model_id)
    print(f'Extracting: model={label}')

    model, tokenizer = load_model(model_id)

    for bi, benchmark in enumerate(BENCHMARKS, 1):
        t0 = time.time()
        print(f'\n[{bi}/{len(BENCHMARKS)}] {benchmark}')
        df = BENCHMARK_LOADERS[benchmark]()
        texts = df['text'].tolist()
        print(f'  Loaded {len(texts)} texts')

        for predictor_name, predictor_cls in PREDICTORS.items():
            predictor = predictor_cls()
            pool_name = predictor.pool.__name__
            cache_path = _feature_path(benchmark, label, pool_name)

            if cache_path.exists():
                print(f'  Cached: {predictor_name} ({cache_path})')
                continue

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tp = time.time()
            predictor.extract_features(model, tokenizer, texts, cache_path=cache_path)
            print(f'  {predictor_name}: {time.time() - tp:.1f}s')

        # Save metadata alongside features.
        meta_path = _meta_path(benchmark, label)
        if not meta_path.exists():
            meta_cols = ['duplicates', 'split']
            if 'orig_idx' in df.columns:
                meta_cols.insert(0, 'orig_idx')
            if 'title' in df.columns:
                meta_cols.insert(0, 'title')
            if 'format' in df.columns:
                meta_cols.append('format')
            df[meta_cols].to_parquet(meta_path)
            print(f'  Saved meta: {meta_path}')

        print(f'  Benchmark total: {time.time() - t0:.1f}s')


# Phase 2: Verify (CPU)


def phase_verify():
    """Check that all expected feature caches exist."""
    missing_features = []
    missing_meta = []
    total_features = 0
    total_meta = 0

    for model_id in PERTURBED_MODELS:
        label = _model_label(model_id)
        for benchmark in BENCHMARKS:
            # Metadata.
            meta_path = _meta_path(benchmark, label)
            if meta_path.exists():
                total_meta += 1
            else:
                missing_meta.append(str(meta_path))

            # Feature files.
            for predictor_cls in PREDICTORS.values():
                pool_name = predictor_cls.pool.__name__
                feat_path = _feature_path(benchmark, label, pool_name)
                if feat_path.exists():
                    total_features += 1
                else:
                    missing_features.append(str(feat_path))

    n_expected_features = len(PERTURBED_MODELS) * len(BENCHMARKS) * len(PREDICTORS)
    n_expected_meta = len(PERTURBED_MODELS) * len(BENCHMARKS)

    print(f'Feature files: {total_features}/{n_expected_features}')
    print(f'Metadata files: {total_meta}/{n_expected_meta}')

    if missing_features or missing_meta:
        if missing_features:
            print(f'\nERROR: {len(missing_features)} missing feature files:')
            for p in sorted(missing_features):
                print(f'  {p}')
        if missing_meta:
            print(f'\nERROR: {len(missing_meta)} missing metadata files:')
            for p in sorted(missing_meta):
                print(f'  {p}')
        sys.exit(1)

    print('\nAll caches present.')


# CLI


def main():
    parser = argparse.ArgumentParser(
        description='Extract hidden-state features across all models and benchmarks')
    sub = parser.add_subparsers(dest='command')

    sp = sub.add_parser('extract', help='Extract hidden-state features (GPU)')
    sp.add_argument('--model', type=int, choices=range(len(PERTURBED_MODELS)),
                    help='Perturbed model index (0-3). Inferred from HUBBLE_TASK_ID if omitted.')

    sub.add_parser('verify', help='Verify all caches are present (CPU)')

    args = parser.parse_args()

    if args.command == 'extract':
        model_idx = args.model
        if model_idx is None:
            task_id = os.environ.get('HUBBLE_TASK_ID')
            if task_id is not None:
                model_idx = int(task_id)
            else:
                parser.error(
                    'Provide --model or set HUBBLE_TASK_ID')
        phase_extract(model_idx)

    elif args.command == 'verify':
        phase_verify()

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
