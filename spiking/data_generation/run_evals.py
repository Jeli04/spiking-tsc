"""Evaluate all 8 Hubble models on all 6 benchmarks.

By default this evaluates any uncached model-benchmark pairs. To evaluate a
single Hubble model, set HUBBLE_TASK_ID to the model index.

Usage:
  uv run python spiking/data_generation/run_evals.py
  HUBBLE_TASK_ID=6 uv run python spiking/data_generation/run_evals.py
"""

import os
from pathlib import Path

import torch

from hubble.data import (
    load_hellaswag_perturbations,
    load_mmlu_perturbations,
    load_piqa_perturbations,
    load_popqa_perturbations,
    load_winogrande_perturbations,
    load_wikipedia_passages,
)
from hubble.eval import evaluate_benchmark_df, load_model
from hubble.runner import HUBBLE_MODELS, _model_label

RESULTS_DIR = Path(__file__).parent / "results"
BENCHMARKS = ["winogrande", "mmlu", "piqa", "popqa", "hellaswag", "wikipedia"]


def load_data(benchmark: str):
    """Load perturbation dataset for the given benchmark."""
    if benchmark == "winogrande":
        df_infill = load_winogrande_perturbations("infill")
        df_mcq = load_winogrande_perturbations("mcq")
        return df_infill, df_mcq
    elif benchmark == "mmlu":
        return load_mmlu_perturbations()
    elif benchmark == "piqa":
        return load_piqa_perturbations()
    elif benchmark == "popqa":
        return load_popqa_perturbations()
    elif benchmark == "hellaswag":
        return load_hellaswag_perturbations()
    elif benchmark == "wikipedia":
        return load_wikipedia_passages()
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def evaluate_model(model_id: str):
    """Evaluate one model on all benchmarks, caching each result."""
    label = _model_label(model_id)

    # Only evaluate missing benchmark caches.
    needed = []
    for benchmark in BENCHMARKS:
        cache_path = RESULTS_DIR / benchmark / f"eval_{label}.parquet"
        if not cache_path.exists():
            needed.append(benchmark)

    if not needed:
        print(f"All benchmarks cached for {label}, skipping.")
        return

    print(f"Loading model: {model_id}")
    model, tokenizer = load_model(model_id)

    for benchmark in needed:
        results_dir = RESULTS_DIR / benchmark
        results_dir.mkdir(exist_ok=True, parents=True)
        cache_path = results_dir / f"eval_{label}.parquet"

        print(f"\nEvaluating {label} on {benchmark}...")
        df = load_data(benchmark)

        # WinoGrande has infill and MCQ splits.
        if benchmark == "winogrande":
            df_infill, df_mcq = df
            df_infill = evaluate_benchmark_df(model, tokenizer, df_infill, label, benchmark)
            df_mcq = evaluate_benchmark_df(model, tokenizer, df_mcq, label, benchmark)
            import pandas as pd
            result_df = pd.concat([df_infill, df_mcq], ignore_index=True)
        else:
            result_df = evaluate_benchmark_df(model, tokenizer, df, label, benchmark)

        result_df.to_parquet(cache_path)
        print(f"Cached {label} ({benchmark}) -> {cache_path}")

    del model, tokenizer
    torch.cuda.empty_cache()


def main():
    task_id = os.environ.get("HUBBLE_TASK_ID")

    if task_id is not None:
        model_id = HUBBLE_MODELS[int(task_id)]
        print(f"HUBBLE_TASK_ID={task_id}: {_model_label(model_id)}")
        evaluate_model(model_id)
    else:
        for model_id in HUBBLE_MODELS:
            evaluate_model(model_id)

    print("\nDone.")


if __name__ == "__main__":
    main()
