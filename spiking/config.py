"""Shared constants for the paper experiments."""

from pathlib import Path

from hubble.benchmarks import (
    BENCHMARK_CACHE_KEYS,
    DOSE_GROUPS,
    MODELS,
)


PAPER_SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PAPER_SRC_DIR.parent
PAPER_RESULTS_DIR = PAPER_SRC_DIR / 'results'


def paper_results_path(*parts: str) -> Path:
    """Return a path rooted at the top-level paper results directory."""
    return PAPER_RESULTS_DIR.joinpath(*parts)

# Benchmarks

BENCHMARKS = (
    'winogrande_mcq',  # paper experiments use the MCQ split
    'mmlu',
    'piqa',
    'hellaswag',
    'popqa',
)
BENCHMARKS_WITH_WIKIPEDIA = BENCHMARKS + ('wikipedia',)

BENCHMARK_LABELS = {
    'winogrande_mcq': 'WinoGrande',
    'mmlu': 'MMLU',
    'piqa': 'PIQA',
    'hellaswag': 'HellaSwag',
    'popqa': 'PopQA',
    'wikipedia': 'Wikipedia',
}

BENCHMARK_BASE_RATES = {
    'winogrande_mcq': 0.5,
    'mmlu': 0.25,
    'piqa': 0.5,
    'hellaswag': 0.25,
    'popqa': 0.0,
}

# Raw MIA score caches still carry both WinoGrande formats.
BENCHMARK_EXP11_MAP = {
    'winogrande_mcq': {'exp11_benchmark': 'winogrande', 'exp11_format': 'mcq'},
    'mmlu': {'exp11_benchmark': 'mmlu', 'exp11_format': None},
    'piqa': {'exp11_benchmark': 'piqa', 'exp11_format': None},
    'popqa': {'exp11_benchmark': 'popqa', 'exp11_format': None},
    'hellaswag': {'exp11_benchmark': 'hellaswag', 'exp11_format': None},
    'wikipedia': {'exp11_benchmark': 'wikipedia', 'exp11_format': None},
}

# Plotting

TEXT_WIDTH = 5.5  # COLM text width in inches

DOSE_LABELS = {'low': 'Low dose', 'mid': 'Mid dose', 'high': 'High dose'}

SAMPLE_EFF_COLORS = {
    'naive': '#69b9a0',
    'ipw': '#8a9bc8',
    'imputation': '#f3d32c',
    'clean_only': '#f4a3a3',
}

# Memorization predictors

ATTACKS = ['loss', 'zlib', 'min_k', 'min_k_plus_plus', 'reference']
HS_PREDICTORS = []  # optional hidden-state predictors
MEM_PREDICTORS = ATTACKS + HS_PREDICTORS

MEM_LABELS = {
    'loss': 'LOSS',
    'zlib': 'Zlib',
    'min_k': 'Min-K%',
    'min_k_plus_plus': 'Min-K%++',
    'reference': 'Reference',
    'final_layer_linear': 'HS mean',
    'final_layer_last_token_linear': 'HS last',
}
MEM_LABELS_TEX = {k: v.replace('%', r'\%') for k, v in MEM_LABELS.items()}

# Correctness predictors

CORR_LABELS = {
    'platt': 'Llama Platt',
    'llama_platt': 'Llama Platt',
    'llm_platt': 'Llama+Platt',
    'roberta': 'RoBERTa',
    'roberta_platt': 'RoBERTa+Platt',
    'pythia_platt': 'Pythia+Platt',
    'qwen_platt': 'Qwen+Platt',
    'uncalibrated': 'Uncalib',
    'isotonic': 'Isotonic',
}

CORR_PREDICTORS_DEFAULT = ['platt', 'roberta']
CORR_PREDICTORS_FULL = [
    'uncalibrated', 'llm_platt', 'isotonic', 'roberta', 'pythia_platt',
]

# External LLM backends shared by confidence extraction and correctness predictors.

EXTERNAL_MODELS = {
    'llama': {
        'sizes': ['8b'],
        'default_size': '8b',
        'model_id': lambda s: 'meta-llama/Llama-3.1-8B',
        'predictor_name': 'llm_platt',
        'c_hat_prefix': 'llama',
        'quality_suffix': '',
        'size_in_quality': False,
    },
    'pythia': {
        'sizes': ['1.4b', '2.8b', '6.9b', '12b'],
        'default_size': '1.4b',
        'model_id': lambda s: f'EleutherAI/pythia-{s}',
        'predictor_name': 'pythia_platt',
        'c_hat_prefix': 'pythia',
        'quality_suffix': 'pythia',
        'size_in_quality': True,
    },
    'qwen': {
        'sizes': ['8b'],
        'default_size': '8b',
        'model_id': lambda s: f'Qwen/Qwen3-{s.upper()}',
        'predictor_name': 'qwen_platt',
        'c_hat_prefix': 'qwen',
        'quality_suffix': 'qwen',
        'size_in_quality': True,
    },
}
PYTHIA_SIZES = EXTERNAL_MODELS['pythia']['sizes']


__all__ = [
    'PAPER_SRC_DIR',
    'PROJECT_ROOT',
    'PAPER_RESULTS_DIR',
    'paper_results_path',
    'BENCHMARK_CACHE_KEYS',
    'DOSE_GROUPS',
    'MODELS',
    'BENCHMARKS',
    'BENCHMARKS_WITH_WIKIPEDIA',
    'BENCHMARK_LABELS',
    'BENCHMARK_BASE_RATES',
    'BENCHMARK_EXP11_MAP',
    'TEXT_WIDTH',
    'DOSE_LABELS',
    'SAMPLE_EFF_COLORS',
    'ATTACKS',
    'HS_PREDICTORS',
    'MEM_PREDICTORS',
    'MEM_LABELS',
    'MEM_LABELS_TEX',
    'CORR_LABELS',
    'CORR_PREDICTORS_DEFAULT',
    'CORR_PREDICTORS_FULL',
    'EXTERNAL_MODELS',
    'PYTHIA_SIZES',
]
