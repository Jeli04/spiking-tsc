"""Evaluate cached correctness predictors together.

Loads cached RoBERTa/external-LLM scores, fits Platt-scaled predictors where
needed, and reports Brier, AUROC, balanced accuracy, bias, and variance.

No GPU needed.

Usage:
  uv run python spiking/correctness/run_evals.py
  uv run python spiking/correctness/run_evals.py --benchmark mmlu
  uv run python spiking/correctness/run_evals.py --pythia-size 6.9b
"""

from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import BENCHMARKS, MODELS
from hubble.corr_predictors import CORRECTNESS_PREDICTORS, LLMConfidencePredictor
from hubble.metrics import metrics_by_dose
from hubble.results import (
    load_cached_confidence,
    load_cached_roberta_scores,
    load_eval_item_pool,
)
from hubble.simulation import stratified_split

try:
    from ._reporting import (
        format_bias_latex_table,
        format_quality_table,
        metrics_to_row as _metrics_to_row,
        print_full_metrics as _print_metrics,
    )
except ImportError:
    from _reporting import (
        format_bias_latex_table,
        format_quality_table,
        metrics_to_row as _metrics_to_row,
        print_full_metrics as _print_metrics,
    )

RESULTS_DIR = Path(__file__).parent / 'results'
CACHE_DIR = Path(__file__).parent / 'cache'
FIGURES_DIR = Path(__file__).parent / 'figures'
PYTHIA_CACHE_DIR = Path(__file__).parent / 'results' / 'pythia_confidence'
QWEN_CACHE_DIR = Path(__file__).parent / 'results' / 'qwen_confidence'
DATA_RESULTS = Path(__file__).resolve().parent.parent / 'data_generation' / 'results'


# Main

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='Re-evaluate all correctness predictors (Llama Platt + RoBERTa) together.')
    parser.add_argument('--model', type=str, default='8b-500b')
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=BENCHMARKS,
                        help='Single benchmark (default: all)')
    parser.add_argument('--perturbed-labels', action='store_true',
                        help='Use perturbed model accuracy as y_clean instead of standard model')
    parser.add_argument('--pythia-size', type=str, default=None,
                        help='Include Pythia+Platt predictor from this Pythia size (e.g. 1.4b, 6.9b)')
    parser.add_argument('--qwen-size', type=str, default=None,
                        help='Include Qwen+Platt predictor from this Qwen size (e.g. 8b)')
    parser.add_argument('--question-only', action='store_true',
                        help='Load question-only RoBERTa scores (from run_roberta.py --question-only)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    quality_rows = []

    label_subdir = 'perturbed_labels' if args.perturbed_labels else 'standard_labels'
    label_variant = 'perturbed' if args.perturbed_labels else 'standard'

    qonly_suffix = '_qonly' if args.question_only else ''

    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)

    for benchmark in benchmarks:
        for model in MODELS:
            if args.model and model != args.model:
                continue

            print(f'\n=== {benchmark} / {model} ===')
            pool = load_eval_item_pool(
                DATA_RESULTS,
                benchmark,
                model,
                confidence_source='perturbed',
                labels_source='perturbed' if args.perturbed_labels else 'standard',
            )
            cal_pool, sim_pool, cal_idx, sim_idx = stratified_split(pool)

            clean_cal_mask = cal_pool.duplicates == 0
            labels_cal_clean = cal_pool.y_clean[clean_cal_mask]
            labels_sim = sim_pool.y_clean

            # Llama predictors (Platt on perturbed confidence)
            if pool.confidence is None:
                print(f'  [SKIP] No perturbed confidence for {benchmark}, skipping Llama predictors')
            else:
                cal_df_clean = pd.DataFrame({
                    'confidence': pool.confidence[cal_idx][clean_cal_mask]})
                sim_df = pd.DataFrame({'confidence': pool.confidence[sim_idx]})

                print(f'  Fitting Llama predictors on {clean_cal_mask.sum()} clean cal items '
                      f'(of {len(cal_idx)} total)')

                for predictor_name, predictor_cls in CORRECTNESS_PREDICTORS.items():
                    predictor = predictor_cls()
                    predictor.fit(cal_df_clean, labels_cal_clean)
                    c_hat_sim = predictor.predict_proba(sim_df)[:, 1]

                    brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                        c_hat_sim,
                        labels_sim,
                        sim_pool.duplicates,
                        confidence=sim_pool.confidence,
                        include_balanced_accuracy=True,
                        include_variance=True,
                    )
                    quality_rows.append({
                        'benchmark': benchmark, 'model': model,
                        'predictor': f'llama_{predictor_name}',
                        **_metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
                    })
                    _print_metrics(f'llama_{predictor_name}', brier, auroc, bal_acc, bias, variance)

            # RoBERTa predictors from cached scores.
            roberta_scores = load_cached_roberta_scores(
                CACHE_DIR,
                benchmark,
                model,
                label_variant=label_variant,
                question_only=args.question_only,
            )

            if roberta_scores is None:
                print(f'  [SKIP] No cached RoBERTa scores for {benchmark} / {model}')
            else:
                print(f'  Loaded cached RoBERTa scores ({len(roberta_scores)} items)')

                # Raw RoBERTa sigmoid scores.
                c_hat_raw = roberta_scores[sim_idx]
                brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                    c_hat_raw,
                    labels_sim,
                    sim_pool.duplicates,
                    confidence=sim_pool.confidence,
                    include_balanced_accuracy=True,
                    include_variance=True,
                )
                roberta_predictor_name = f'roberta{qonly_suffix}'
                quality_rows.append({
                    'benchmark': benchmark, 'model': model,
                    'predictor': roberta_predictor_name,
                    **_metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
                })
                _print_metrics(roberta_predictor_name, brier, auroc, bal_acc, bias, variance)

            # Pythia + Platt from cached confidence.
            if args.pythia_size:
                pythia_label = f'pythia-{args.pythia_size}'
                pythia_conf = load_cached_confidence(
                    benchmark,
                    PYTHIA_CACHE_DIR,
                    pythia_label,
                    suffix=qonly_suffix,
                )
                if pythia_conf is None:
                    print(f'  [SKIP] No cached Pythia confidence for {benchmark} at size {args.pythia_size}')
                else:
                    print(f'  Loaded cached Pythia confidence ({len(pythia_conf)} items)')

                    cal_conf = pythia_conf[cal_idx][clean_cal_mask]
                    sim_conf = pythia_conf[sim_idx]

                    predictor = LLMConfidencePredictor(f'EleutherAI/{pythia_label}')
                    predictor.fit(cal_conf, labels_cal_clean)
                    c_hat_platt = predictor.predict_proba(sim_conf)[:, 1]

                    brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                        c_hat_platt,
                        labels_sim,
                        sim_pool.duplicates,
                        confidence=sim_pool.confidence,
                        include_balanced_accuracy=True,
                        include_variance=True,
                    )
                    quality_rows.append({
                        'benchmark': benchmark, 'model': model,
                        'predictor': f'pythia_platt_{args.pythia_size}',
                        **_metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
                    })
                    _print_metrics(f'pythia_platt_{args.pythia_size}', brier, auroc, bal_acc, bias, variance)

            # Qwen + Platt from cached confidence.
            if args.qwen_size:
                qwen_label = f'Qwen3-{args.qwen_size.upper()}'
                qwen_conf = load_cached_confidence(
                    benchmark,
                    QWEN_CACHE_DIR,
                    qwen_label,
                    suffix=qonly_suffix,
                )
                if qwen_conf is None:
                    print(f'  [SKIP] No cached Qwen confidence for {benchmark} at size {args.qwen_size}')
                else:
                    print(f'  Loaded cached Qwen confidence ({len(qwen_conf)} items)')

                    cal_conf = qwen_conf[cal_idx][clean_cal_mask]
                    sim_conf = qwen_conf[sim_idx]

                    predictor = LLMConfidencePredictor(f'Qwen/{qwen_label}')
                    predictor.fit(cal_conf, labels_cal_clean)
                    c_hat_platt = predictor.predict_proba(sim_conf)[:, 1]

                    brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                        c_hat_platt,
                        labels_sim,
                        sim_pool.duplicates,
                        confidence=sim_pool.confidence,
                        include_balanced_accuracy=True,
                        include_variance=True,
                    )
                    quality_rows.append({
                        'benchmark': benchmark, 'model': model,
                        'predictor': f'qwen_platt_{args.qwen_size}',
                        **_metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
                    })
                    _print_metrics(f'qwen_platt_{args.qwen_size}', brier, auroc, bal_acc, bias, variance)

    # Save results under the label variant.
    results_subdir = RESULTS_DIR / label_subdir
    results_subdir.mkdir(parents=True, exist_ok=True)
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(results_subdir / f'eval_quality{qonly_suffix}.csv', index=False)
    quality_df.to_parquet(results_subdir / f'eval_quality{qonly_suffix}.parquet')

    # Write summary tables.
    figures_subdir = FIGURES_DIR / label_subdir
    figures_subdir.mkdir(parents=True, exist_ok=True)

    md = format_quality_table(quality_df)
    md_path = figures_subdir / f'eval_table{qonly_suffix}.md'
    md_path.write_text(md)
    print(f'\n{md}')
    print(f'\nSaved: {md_path}')

    # Write LaTeX bias table.
    tex = format_bias_latex_table(quality_df)
    tex_path = figures_subdir / f'abs_bias_table{qonly_suffix}.tex'
    tex_path.write_text(tex)
    print(f'\nSaved: {tex_path}')


if __name__ == '__main__':
    main()
