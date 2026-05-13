"""Generate c_hat predictions with an external LLM.

Fits two predictors per (benchmark, model) pair:
  - ConfidencePredictor ("platt"): Platt scaling on the perturbed model's own confidence.
  - LLMConfidencePredictor ("<external>_platt"): Platt scaling on an external LLM's
    confidence (llama / pythia / qwen).

External confidence comes from the shared confidence cache. Run extraction
first::

    sbatch slurm/run_gpu.sbatch src/spiking/data_generation/run_llm_confidence.py \
        extract --external {llama,pythia,qwen} [--size SIZE]

Fits on clean cal-split items, predicts c_hat on all sim-split items.

Usage:
  # Llama-3.1-8B.
  uv run python src/spiking/correctness/run_external_llm.py --external llama

  # Pythia at a specific size.
  uv run python src/spiking/correctness/run_external_llm.py --external pythia --size 1.4b

  # Qwen.
  uv run python src/spiking/correctness/run_external_llm.py --external qwen --size 8b

  # Restrict to one benchmark or use perturbed labels.
  uv run python src/spiking/correctness/run_external_llm.py --external llama --benchmark mmlu
  uv run python src/spiking/correctness/run_external_llm.py --external pythia --size 1.4b --perturbed-labels
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import BENCHMARKS, EXTERNAL_MODELS, MODELS
from hubble.corr_predictors import ConfidencePredictor, LLMConfidencePredictor
from hubble.metrics import metrics_by_dose
from hubble.results import load_cached_confidence, load_eval_item_pool
from hubble.simulation import stratified_split

try:
    from ._reporting import format_quality_table, metrics_to_row, print_full_metrics
except ImportError:
    from _reporting import format_quality_table, metrics_to_row, print_full_metrics

BASE_DIR = Path(__file__).parent
DATA_RESULTS = Path(__file__).resolve().parent.parent / 'data_generation' / 'results'
CONFIDENCE_DIR = DATA_RESULTS / 'confidence'


def _output_dirs(perturbed_labels):
    suffix = 'perturbed_labels' if perturbed_labels else 'standard_labels'
    return BASE_DIR / 'results' / suffix, BASE_DIR / 'figures' / suffix


def _load_external_confidence(benchmark, label):
    """Load external LLM confidence from the shared cache."""
    confidence = load_cached_confidence(benchmark, CONFIDENCE_DIR, label)
    if confidence is None:
        expected = CONFIDENCE_DIR / benchmark / f'confidence_{label}.parquet'
        print(f'  [llm_platt] Cache not found for {benchmark}: {expected}')
        print(f'  Run: uv run python src/spiking/data_generation/run_llm_confidence.py '
              f'extract --external <backend> [--size <size>]')
        return None
    return confidence


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--external', required=True,
                        choices=list(EXTERNAL_MODELS.keys()),
                        help='External LLM backend')
    parser.add_argument('--size', type=str, default=None,
                        help='Model size (backend-specific, default varies)')
    parser.add_argument('--model', type=str, default='8b-500b',
                        help='Hubble model to evaluate (default: 8b-500b)')
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=BENCHMARKS,
                        help='Single benchmark (default: all)')
    parser.add_argument('--perturbed-labels', action='store_true',
                        help='Use perturbed model accuracy as y_clean instead of standard model')
    args = parser.parse_args()

    cfg = EXTERNAL_MODELS[args.external]
    size = args.size or cfg['default_size']
    if size not in cfg['sizes']:
        parser.error(f"--size must be one of {cfg['sizes']} for --external {args.external}")

    model_id = cfg['model_id'](size)
    label = model_id.split('/')[-1]
    predictor_name = cfg['predictor_name']

    results_dir, figures_dir = _output_dirs(args.perturbed_labels)
    results_dir.mkdir(parents=True, exist_ok=True)

    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    quality_rows = []

    for benchmark in benchmarks:
        for model in MODELS:
            if args.model and model != args.model:
                continue

            print(f'\n=== {benchmark} / {model} (external: {label}) ===')
            pool = load_eval_item_pool(
                DATA_RESULTS,
                benchmark,
                model,
                confidence_source='perturbed',
                labels_source='perturbed' if args.perturbed_labels else 'standard',
            )
            cal_pool, sim_pool, cal_idx, sim_idx = stratified_split(pool)

            # Fit on clean cal items, where perturbed and standard labels agree.
            clean_cal_mask = cal_pool.duplicates == 0
            labels_cal_clean = cal_pool.y_clean[clean_cal_mask]
            labels_sim = sim_pool.y_clean

            print(f'  Fitting on {clean_cal_mask.sum()} clean cal items '
                  f'(of {len(cal_idx)} total)')

            c_hat_dict = {}

            def _run_predictor(name, predictor, cal_conf_clean, sim_conf):
                cal_df = pd.DataFrame({'confidence': cal_conf_clean})
                sim_df = pd.DataFrame({'confidence': sim_conf})
                predictor.fit(cal_df, labels_cal_clean)
                c_hat_sim = predictor.predict_proba(sim_df)[:, 1]
                c_hat_dict[name] = c_hat_sim

                brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                    c_hat_sim,
                    labels_sim,
                    sim_pool.duplicates,
                    confidence=sim_pool.confidence,
                    include_balanced_accuracy=True,
                    include_variance=True,
                )
                extra = {f'{args.external}_size': size} if cfg['size_in_quality'] else {}
                quality_rows.append({
                    'benchmark': benchmark,
                    'model': model,
                    'predictor': name,
                    **metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
                    **extra,
                })
                print_full_metrics(name, brier, auroc, bal_acc, bias, variance)

            # Perturbed model confidence.
            _run_predictor('platt', ConfidencePredictor(),
                       pool.confidence[cal_idx][clean_cal_mask],
                       pool.confidence[sim_idx])

            # External LLM confidence.
            ext_conf = _load_external_confidence(benchmark, label)
            if ext_conf is not None:
                _run_predictor(predictor_name, LLMConfidencePredictor(model_id),
                           ext_conf[cal_idx][clean_cal_mask],
                           ext_conf[sim_idx])

            # Save c_hat arrays for simulation.
            out_dir = results_dir / benchmark
            out_dir.mkdir(parents=True, exist_ok=True)
            if cfg['size_in_quality']:
                npz_name = f'c_hat_{cfg["c_hat_prefix"]}_{label}_{model}.npz'
            else:
                npz_name = f'c_hat_{cfg["c_hat_prefix"]}_{model}.npz'
            np.savez(out_dir / npz_name, **c_hat_dict)

    # Save quality table.
    quality_df = pd.DataFrame(quality_rows)
    if cfg['quality_suffix']:
        q_stem = f'predictor_quality_{cfg["quality_suffix"]}_{label}'
        md_stem = f'brier_table_{cfg["quality_suffix"]}_{label}'
    else:
        q_stem = 'predictor_quality'
        md_stem = 'brier_table'

    quality_df.to_csv(results_dir / f'{q_stem}.csv', index=False)
    quality_df.to_parquet(results_dir / f'{q_stem}.parquet')

    figures_dir.mkdir(parents=True, exist_ok=True)
    md = format_quality_table(quality_df)
    md_path = figures_dir / f'{md_stem}.md'
    md_path.write_text(md)
    print(f'\n{md}')
    print(f'\nSaved: {md_path}')


if __name__ == '__main__':
    main()
