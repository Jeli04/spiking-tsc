"""Unified evaluation runner for Hubble benchmark experiments."""

import os
from pathlib import Path
import pandas as pd
import torch

from hubble.eval import load_model

# The 8 core Hubble models (2x2x2: size x tokens x condition).
HUBBLE_MODELS = [
    "allegrolab/hubble-1b-100b_toks-standard-hf",
    "allegrolab/hubble-1b-100b_toks-perturbed-hf",
    "allegrolab/hubble-1b-500b_toks-standard-hf",
    "allegrolab/hubble-1b-500b_toks-perturbed-hf",
    "allegrolab/hubble-8b-100b_toks-standard-hf",
    "allegrolab/hubble-8b-100b_toks-perturbed-hf",
    "allegrolab/hubble-8b-500b_toks-standard-hf",
    "allegrolab/hubble-8b-500b_toks-perturbed-hf",
]


def _model_label(model_id: str) -> str:
    """Derive label from HuggingFace model ID, e.g. 'allegrolab/hubble-1b-...' -> 'hubble-1b-...'."""
    return model_id.split("/")[-1]

def run_eval(
    results_dir: Path,
    load_data,
    eval_fn,
    models: list[str] | None = None,
):
    """Run the standard Hubble evaluation loop.

    Handles SLURM array dispatch, caching to parquet, and GPU cleanup.

    Args:
        results_dir: Where to write eval_{label}.parquet files.
        load_data: Returns the benchmark DataFrame.
        eval_fn: Signature (model, tokenizer, df, label) -> df.
        models: HuggingFace model IDs. Defaults to HUBBLE_MODELS.
    """
    if models is None:
        models = HUBBLE_MODELS
    results_dir.mkdir(exist_ok=True, parents=True)

    df = load_data()
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")

    if task_id is not None:
        idx = int(task_id)
        model_id = models[idx]
        label = _model_label(model_id)
        print(f"Array task {idx}: evaluating {label}")
        _evaluate_single(df, model_id, eval_fn, results_dir)
    else:
        for model_id in models:
            label = _model_label(model_id)
            cache_path = results_dir / f"eval_{label}.parquet"
            if not cache_path.exists():
                _evaluate_single(df, model_id, eval_fn, results_dir)
                torch.cuda.empty_cache()


def _evaluate_single(
    df: pd.DataFrame,
    model_id: str,
    eval_fn,
    results_dir: Path,
):
    """Evaluate one model and cache the result as parquet."""
    label = _model_label(model_id)
    cache_path = results_dir / f"eval_{label}.parquet"
    if cache_path.exists():
        print(f"Cache exists for {label}, skipping.")
        return

    print(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id)

    print(f"Evaluating {len(df)} examples on {label}...")
    result_df = eval_fn(model, tokenizer, df, label)

    result_df.to_parquet(cache_path)
    print(f"Cached {label} → {cache_path}")


