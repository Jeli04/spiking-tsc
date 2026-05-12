"""Shared benchmark metadata for the paper release layer.

This module intentionally centralizes metadata that was previously duplicated
across many paper scripts: benchmark labels, cache-key mappings, default model
tags, and the paper's dose-group definitions.
"""

from __future__ import annotations

from hubble.data import BENCHMARK_LOADERS, QUESTION_COLUMNS


BENCHMARKS = (
    'winogrande_infill',
    'winogrande_mcq',
    'mmlu',
    'piqa',
    'hellaswag',
    'popqa',
)

BENCHMARKS_WITH_WIKIPEDIA = BENCHMARKS + ('wikipedia',)

MODELS = ('8b-500b',)

# Paper-facing dose groups. Note that these differ slightly from the generic
# simulation defaults in hubble.simulation.
DOSE_GROUPS = {
    'low': [1],
    'mid': [4, 16],
    'high': [64, 256],
}

BENCHMARK_LABELS_SHORT = {
    'winogrande_infill': 'WG (infill)',
    'winogrande_mcq': 'WG (mcq)',
    'hellaswag': 'HellaSwag',
    'mmlu': 'MMLU',
    'piqa': 'PIQA',
    'popqa': 'PopQA',
    'wikipedia': 'Wikipedia',
}

BENCHMARK_LABELS_LONG = {
    'winogrande_infill': 'WinoGrande (infill)',
    'winogrande_mcq': 'WinoGrande (MCQ)',
    'hellaswag': 'HellaSwag',
    'mmlu': 'MMLU',
    'piqa': 'PIQA',
    'popqa': 'PopQA',
    'wikipedia': 'Wikipedia',
}

BENCHMARK_BASE_RATES = {
    'winogrande_infill': 0.5,
    'winogrande_mcq': 0.5,
    'hellaswag': 0.25,
    'mmlu': 0.25,
    'piqa': 0.5,
    'popqa': 0.0,
}

# Cache-key mapping for assets that store the two WinoGrande formats together.
BENCHMARK_CACHE_KEYS = {
    'winogrande_infill': ('winogrande', 'infill'),
    'winogrande_mcq': ('winogrande', 'mcq'),
    'mmlu': ('mmlu', None),
    'piqa': ('piqa', None),
    'hellaswag': ('hellaswag', None),
    'popqa': ('popqa', None),
}


def benchmark_label(name: str, *, style: str = 'short') -> str:
    """Return the display label for a benchmark."""
    if style == 'short':
        return BENCHMARK_LABELS_SHORT.get(name, name)
    if style == 'long':
        return BENCHMARK_LABELS_LONG.get(name, name)
    raise ValueError(f'Unknown label style: {style}')


def cache_key_and_format(name: str) -> tuple[str, str | None]:
    """Return (cache_key, format_filter) for a benchmark."""
    if name not in BENCHMARK_CACHE_KEYS:
        raise KeyError(f'No cache mapping registered for benchmark: {name}')
    return BENCHMARK_CACHE_KEYS[name]


__all__ = [
    'BENCHMARKS',
    'BENCHMARKS_WITH_WIKIPEDIA',
    'MODELS',
    'DOSE_GROUPS',
    'BENCHMARK_LABELS_SHORT',
    'BENCHMARK_LABELS_LONG',
    'BENCHMARK_BASE_RATES',
    'BENCHMARK_CACHE_KEYS',
    'BENCHMARK_LOADERS',
    'QUESTION_COLUMNS',
    'benchmark_label',
    'cache_key_and_format',
]
