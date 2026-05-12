"""Helpers for loading paper evaluation artifacts and aligning cached outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hubble.benchmarks import BENCHMARK_CACHE_KEYS
from hubble.data import BENCHMARK_LOADERS, QUESTION_COLUMNS
from hubble.simulation import ItemPool, stratified_split


def model_tag(model: str) -> str:
    """Return the Hubble file tag for a short model name like ``8b-500b``."""
    return f'hubble-{model}_toks'


def eval_parquet_paths(eval_root: str | Path, benchmark: str, model: str) -> tuple[Path, Path]:
    """Return (standard_path, perturbed_path) for a benchmark/model pair."""
    root = Path(eval_root)
    tag = model_tag(model)
    bench_dir = root / benchmark
    return (
        bench_dir / f'eval_{tag}-standard-hf.parquet',
        bench_dir / f'eval_{tag}-perturbed-hf.parquet',
    )


def load_eval_frames(eval_root: str | Path, benchmark: str, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load standard and perturbed eval parquets for a benchmark/model pair."""
    std_path, prt_path = eval_parquet_paths(eval_root, benchmark, model)
    return pd.read_parquet(std_path), pd.read_parquet(prt_path)


def accuracy_prefix(std_df: pd.DataFrame, model: str) -> str:
    """Infer whether the eval parquet stores ``acc_`` or ``exact_match_``."""
    tag = model_tag(model)
    return 'acc' if f'acc_{tag}-standard-hf' in std_df.columns else 'exact_match'


def load_eval_item_pool(
    eval_root: str | Path,
    benchmark: str,
    model: str,
    *,
    confidence_source: str | None = 'standard',
    labels_source: str = 'standard',
) -> ItemPool:
    """Build an ItemPool from eval parquets with configurable label/confidence source.

    Args:
        eval_root: Root containing per-benchmark eval parquet directories.
        benchmark: Benchmark name.
        model: Short model tag like ``8b-500b``.
        confidence_source: ``'standard'``, ``'perturbed'``, or ``None``.
        labels_source: ``'standard'`` or ``'perturbed'`` for the clean target.
    """
    if labels_source not in {'standard', 'perturbed'}:
        raise ValueError(f'Unknown labels_source: {labels_source}')
    if confidence_source not in {'standard', 'perturbed', None}:
        raise ValueError(f'Unknown confidence_source: {confidence_source}')

    tag = model_tag(model)
    std_path, prt_path = eval_parquet_paths(eval_root, benchmark, model)
    std_df, prt_df = load_eval_frames(eval_root, benchmark, model)
    acc_prefix = accuracy_prefix(std_df, model)

    acc_clean_col = f'{acc_prefix}_{tag}-{labels_source}-hf'

    if confidence_source == 'standard':
        confidence_col = f'confidence_{tag}-standard-hf'
        pool = ItemPool.from_eval_parquets(
            str(std_path),
            str(prt_path),
            acc_clean_col=acc_clean_col,
            acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
            confidence_col=confidence_col if confidence_col in std_df.columns else None,
        )
    else:
        pool = ItemPool.from_eval_parquets(
            str(std_path),
            str(prt_path),
            acc_clean_col=acc_clean_col,
            acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
            confidence_col=None,
        )
        if confidence_source == 'perturbed':
            confidence_col = f'confidence_{tag}-perturbed-hf'
            if confidence_col in prt_df.columns:
                pool.confidence = prt_df[confidence_col].values

    return pool


def load_sim_item_pool(
    eval_root: str | Path,
    benchmark: str,
    model: str,
    *,
    confidence_source: str | None = 'standard',
    labels_source: str = 'standard',
    seed: int = 42,
) -> ItemPool:
    """Return only the simulation half of the eval ItemPool."""
    full_pool = load_eval_item_pool(
        eval_root,
        benchmark,
        model,
        confidence_source=confidence_source,
        labels_source=labels_source,
    )
    _, sim_pool, _, _ = stratified_split(full_pool, seed=seed)
    return sim_pool


def align_texts_to_eval_rows(
    eval_root: str | Path,
    benchmark: str,
    model: str,
    *,
    question_only: bool = False,
) -> list[str]:
    """Return benchmark texts aligned with eval-parquet row ordering."""
    std_path, _ = eval_parquet_paths(eval_root, benchmark, model)
    eval_df = pd.read_parquet(std_path, columns=['orig_idx'])
    text_df = BENCHMARK_LOADERS[benchmark]()
    text_col = QUESTION_COLUMNS[benchmark] if question_only else 'text'
    merged = eval_df.merge(text_df[['orig_idx', text_col]], on='orig_idx', how='left')
    if not merged[text_col].notna().all():
        raise RuntimeError(f'Missing {text_col} for some orig_idx in {benchmark}')
    return merged[text_col].tolist()


def roberta_score_cache_path(
    cache_root: str | Path,
    benchmark: str,
    model: str,
    *,
    label_variant: str = 'standard',
    question_only: bool = False,
) -> Path:
    """Return the canonical cache path for raw RoBERTa correctness scores."""
    if label_variant not in {'standard', 'perturbed'}:
        raise ValueError(f'Unknown label_variant: {label_variant}')

    suffix = '_qonly' if question_only else ''
    return Path(cache_root) / label_variant / f'scores_{benchmark}_{model}{suffix}.npz'


def load_cached_roberta_scores(
    cache_root: str | Path,
    benchmark: str,
    model: str,
    *,
    label_variant: str = 'standard',
    question_only: bool = False,
) -> np.ndarray | None:
    """Load canonical raw RoBERTa correctness scores, if present."""
    path = roberta_score_cache_path(
        cache_root,
        benchmark,
        model,
        label_variant=label_variant,
        question_only=question_only,
    )
    if not path.exists():
        return None
    return np.load(path)['scores']


def build_cache_dataframe(cache_key: str) -> pd.DataFrame:
    """Load the full DataFrame used to build a benchmark cache key."""
    return BENCHMARK_LOADERS[cache_key]()


def load_cached_confidence(
    benchmark: str,
    cache_root: str | Path,
    label: str,
    *,
    suffix: str = '',
) -> np.ndarray | None:
    """Load cached confidence aligned to a benchmark, applying format filters."""
    if benchmark not in BENCHMARK_CACHE_KEYS:
        return None

    cache_key, format_filter = BENCHMARK_CACHE_KEYS[benchmark]
    cache_root = Path(cache_root)
    conf_path = cache_root / cache_key / f'confidence_{label}{suffix}.parquet'
    if not conf_path.exists():
        return None

    conf_df = pd.read_parquet(conf_path)
    if format_filter is None:
        return conf_df['confidence'].values

    meta_path = cache_root / cache_key / 'meta.parquet'
    if not meta_path.exists():
        return conf_df['confidence'].values

    meta = pd.read_parquet(meta_path)
    mask = meta['format'] == format_filter
    return conf_df.loc[mask, 'confidence'].values


__all__ = [
    'model_tag',
    'eval_parquet_paths',
    'load_eval_frames',
    'accuracy_prefix',
    'load_eval_item_pool',
    'load_sim_item_pool',
    'align_texts_to_eval_rows',
    'roberta_score_cache_path',
    'load_cached_roberta_scores',
    'build_cache_dataframe',
    'load_cached_confidence',
]
