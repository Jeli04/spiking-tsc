"""Extract external LLM confidence scores across all benchmarks.

Supports multiple external backends (llama / pythia / qwen). The extracted
confidence parquet files are written to::

    src/spiking/data_generation/results/confidence/{cache_key}/confidence_{label}.parquet

which is where correctness/run_external_llm.py expects to find them.

Two-phase workflow:
  Phase 1 (extract): GPU. Evaluate external LLM on benchmarks, cache confidence.
  Phase 2 (verify):  CPU. Check all expected caches exist.

Usage:
  # Extract confidence for all benchmarks with the default Llama model.
  uv run python src/spiking/data_generation/run_llm_confidence.py extract --external llama

  # Pythia at a specific size.
  sbatch slurm/run_gpu.sbatch src/spiking/data_generation/run_llm_confidence.py \
      extract --external pythia --size 1.4b

  # Qwen, downloaded to <project_root>/models/.
  sbatch slurm/run_gpu.sbatch src/spiking/data_generation/run_llm_confidence.py \
      extract --external qwen --size 8b --local-models

  # Restrict to a subset of benchmarks.
  uv run python src/spiking/data_generation/run_llm_confidence.py \
      extract --external pythia --size 1.4b --benchmark mmlu piqa

  # Verify all caches for a backend.
  uv run python src/spiking/data_generation/run_llm_confidence.py verify --external llama
"""

import argparse
import sys
import time
from pathlib import Path

from spiking.config import EXTERNAL_MODELS
from hubble.data import BENCHMARK_LOADERS
from hubble.corr_probes import LLMConfidenceProbe

# Constants

RESULTS_DIR = Path(__file__).parent / 'results'
CONFIDENCE_DIR = RESULTS_DIR / 'confidence'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
LOCAL_MODELS_DIR = PROJECT_ROOT / 'models'

# BENCHMARK_LOADERS uses the raw Hubble cache keys.
BENCHMARKS = ['winogrande', 'mmlu', 'piqa', 'popqa', 'hellaswag']


def _resolve_external(external: str, size: str | None) -> tuple[str, str]:
    cfg = EXTERNAL_MODELS[external]
    size = size or cfg['default_size']
    if size not in cfg['sizes']:
        raise SystemExit(
            f"--size must be one of {cfg['sizes']} for --external {external}")
    model_id = cfg['model_id'](size)
    label = model_id.split('/')[-1]
    return model_id, label


def _resolve_model_path(model_id: str, local_models: bool) -> str:
    """Return the model path, downloading locally when requested."""
    if not local_models:
        return model_id
    from huggingface_hub import snapshot_download
    local_path = LOCAL_MODELS_DIR / model_id.split('/')[-1]
    if not local_path.exists():
        print(f'  Downloading {model_id} -> {local_path}')
        snapshot_download(repo_id=model_id, local_dir=str(local_path))
    else:
        print(f'  Using local model: {local_path}')
    return str(local_path)


def _confidence_path(benchmark: str, label: str) -> Path:
    return CONFIDENCE_DIR / benchmark / f'confidence_{label}.parquet'


def _meta_path(benchmark: str) -> Path:
    return CONFIDENCE_DIR / benchmark / 'meta.parquet'


# Phase 1: Extract (GPU)


def phase_extract(
    external: str,
    size: str | None,
    benchmarks: list[str] | None,
    local_models: bool,
):
    """Extract LLM confidence scores for specified benchmarks."""
    from hubble.eval import load_model  # deferred import (needs GPU)

    model_id, label = _resolve_external(external, size)
    benchmarks = benchmarks or BENCHMARKS

    probe = LLMConfidenceProbe(model_id)
    print(f'Extracting confidence: backend={external} model={label}')

    load_path = _resolve_model_path(model_id, local_models)
    model, tokenizer = load_model(load_path)

    for bi, benchmark in enumerate(benchmarks, 1):
        t0 = time.time()
        print(f'\n[{bi}/{len(benchmarks)}] {benchmark}')
        df = BENCHMARK_LOADERS[benchmark]()
        print(f'  Loaded {len(df)} examples')

        cache_path = _confidence_path(benchmark, label)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        conf = probe.extract_confidence(
            model, tokenizer, df, benchmark, cache_path=cache_path,
        )

        meta_path = _meta_path(benchmark)
        if not meta_path.exists():
            meta_cols = ['orig_idx', 'duplicates', 'split']
            if 'format' in df.columns:
                meta_cols.append('format')
            df[meta_cols].to_parquet(meta_path)
            print(f'  Saved meta: {meta_path}')

        print(f'  Extracted {len(conf)} confidence scores')
        print(f'  Benchmark total: {time.time() - t0:.1f}s')


# Phase 2: Verify (CPU)


def phase_verify(external: str, size: str | None):
    """Check that all expected confidence caches exist for a backend."""
    _, label = _resolve_external(external, size)
    missing = []
    total = 0

    for benchmark in BENCHMARKS:
        conf_path = _confidence_path(benchmark, label)
        if conf_path.exists():
            total += 1
        else:
            missing.append(str(conf_path))

        meta_path = _meta_path(benchmark)
        if meta_path.exists():
            total += 1
        else:
            missing.append(str(meta_path))

    n_expected = len(BENCHMARKS) * 2  # confidence and metadata
    print(f'[{label}] Cache files: {total}/{n_expected}')

    if missing:
        print(f'\nERROR: {len(missing)} missing files:')
        for p in sorted(missing):
            print(f'  {p}')
        sys.exit(1)

    print('\nAll caches present.')


# CLI


def main():
    parser = argparse.ArgumentParser(
        description='Extract external LLM confidence scores across all benchmarks')
    parser.add_argument('command', choices=['extract', 'verify'])
    parser.add_argument('--external', required=True,
                        choices=list(EXTERNAL_MODELS.keys()),
                        help='External LLM backend')
    parser.add_argument('--size', type=str, default=None,
                        help='Model size (backend-specific, default varies)')
    parser.add_argument('--benchmark', type=str, choices=BENCHMARKS,
                        nargs='+', help='Benchmark(s) to extract. Defaults to all.')
    parser.add_argument('--local-models', action='store_true',
                        help='Download and load models from <project_root>/models/ '
                             'instead of the HF cache')
    args = parser.parse_args()

    if args.command == 'extract':
        phase_extract(args.external, args.size, args.benchmark, args.local_models)
    else:
        phase_verify(args.external, args.size)


if __name__ == '__main__':
    main()
