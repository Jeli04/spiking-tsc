"""Train per-benchmark RoBERTa correctness predictors and generate c_hat predictions.

For each benchmark, fine-tunes RoBERTa on clean calibration items (text -> correctness),
extracts scores, and uses sigmoid probabilities directly as c_hat on simulation items.
Reports Brier score + AUROC by dose group.

With --platt, loads cached raw RoBERTa scores and applies Platt scaling.

GPU required for default (finetune) mode.

Usage:
  sbatch slurm/run_gpu.sbatch src/spiking/correctness/run_roberta.py
  uv run python src/spiking/correctness/run_roberta.py --benchmark mmlu
  uv run python src/spiking/correctness/run_roberta.py --benchmark mmlu --epochs 1
  uv run python src/spiking/correctness/run_roberta.py --platt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from spiking.config import BENCHMARKS, MODELS
from hubble.corr_predictors import RoBERTaCorrectnessPredictor
from hubble.metrics import metrics_by_dose
from hubble.results import (
    align_texts_to_eval_rows,
    load_cached_roberta_scores,
    load_eval_item_pool,
    roberta_score_cache_path,
)
from hubble.simulation import stratified_split

try:
    from ._reporting import (
        format_quality_table,
        metrics_to_row,
        print_full_metrics,
    )
except ImportError:
    from _reporting import (
        format_quality_table,
        metrics_to_row,
        print_full_metrics,
    )

RESULTS_DIR = Path(__file__).parent / 'results'
CACHE_DIR = Path(__file__).parent / 'cache'
FIGURES_DIR = Path(__file__).parent / 'figures'
DATA_RESULTS = Path(__file__).resolve().parent.parent / 'data_generation' / 'results'


# Main

def main():
    parser = argparse.ArgumentParser(
        description='Train per-benchmark RoBERTa correctness predictors and generate c_hat predictions.'
    )
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=BENCHMARKS,
                        help='Single benchmark (default: all)')
    parser.add_argument('--platt', action='store_true',
                        help='Platt scaling mode: skip finetuning, load cached raw RoBERTa '
                             'scores from a previous non-Platt run, and fit Platt scaling '
                             'on clean cal items. CPU only.')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max-length', type=int, default=512)
    parser.add_argument('--perturbed-labels', action='store_true',
                        help='Use perturbed model accuracy as y_clean instead of standard model')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--classifier-dropout', type=float, default=0.1,
                        help='Dropout probability for the classification head. Default: 0.1')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0=none, e.g. 0.1 maps 0/1 to 0.05/0.95). Default: 0.0')
    parser.add_argument('--freeze-layers', type=int, nargs=2, default=None,
                        metavar=('START', 'END'),
                        help='Freeze encoder layers in range [START, END). '
                             'E.g. --freeze-layers 0 10 freezes layers 0-9. Default: None')
    parser.add_argument('--question-only', action='store_true',
                        help='Use question/prompt text only (without the answer) as input. '
                             'Default: use the full text including the answer.')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    quality_rows = []

    label_variant = 'perturbed' if args.perturbed_labels else 'standard'
    cache_dir = CACHE_DIR / label_variant

    qonly_suffix = '_qonly' if args.question_only else ''

    benchmarks = [args.benchmark] if args.benchmark else list(BENCHMARKS)
    predictor_name = ('roberta_platt' if args.platt else 'roberta') + qonly_suffix

    for benchmark in benchmarks:
        for model in MODELS:
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

            model_dir = cache_dir / f'roberta_{benchmark}_{model}{qonly_suffix}'
            scores_cache = roberta_score_cache_path(
                CACHE_DIR,
                benchmark,
                model,
                label_variant=label_variant,
                question_only=args.question_only,
            )

            if args.platt:
                # Platt mode uses cached raw scores.
                all_scores = load_cached_roberta_scores(
                    CACHE_DIR,
                    benchmark,
                    model,
                    label_variant=label_variant,
                    question_only=args.question_only,
                )
                assert all_scores is not None, (
                    f'Cached RoBERTa scores not found at {scores_cache}. '
                    f'Run this script once without --platt first.')
                print(f'  Loaded cached RoBERTa scores ({len(all_scores)} items)')

                cal_scores = all_scores[cal_idx][clean_cal_mask]
                sim_scores = all_scores[sim_idx]

                predictor = RoBERTaCorrectnessPredictor(model_dir=model_dir)
                predictor.fit(cal_scores, labels_cal_clean)
                c_hat_sim = predictor.predict_proba(sim_scores)[:, 1]

                print(f'  Platt scaling on {clean_cal_mask.sum()} clean cal items '
                      f'(of {len(cal_idx)} total)')

            else:
                # Default mode fine-tunes RoBERTa and uses sigmoid scores.
                texts = align_texts_to_eval_rows(
                    DATA_RESULTS,
                    benchmark,
                    model,
                    question_only=args.question_only,
                )

                train_texts = [texts[i] for i in cal_idx[clean_cal_mask]]
                train_labels = labels_cal_clean.tolist()

                print(f'  Training on {len(train_texts)} clean cal items '
                      f'(of {len(cal_idx)} total)')

                predictor = RoBERTaCorrectnessPredictor(
                    model_dir=model_dir, max_length=args.max_length)
                predictor.train_model(
                    train_texts, train_labels,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    seed=args.seed,
                    classifier_dropout=args.classifier_dropout,
                    label_smoothing=args.label_smoothing,
                    freeze_layers=tuple(args.freeze_layers) if args.freeze_layers else None,
                )

                all_scores = predictor.extract_scores(
                    texts, batch_size=args.batch_size * 2,
                    cache_path=scores_cache,
                )
                c_hat_sim = all_scores[sim_idx]

            brier, auroc, bal_acc, bias, variance = metrics_by_dose(
                c_hat_sim,
                labels_sim,
                sim_pool.duplicates,
                confidence=sim_pool.confidence,
                include_balanced_accuracy=True,
                include_variance=True,
            )

            quality_rows.append({
                'benchmark': benchmark,
                'model': model,
                'predictor': predictor_name,
                **metrics_to_row(brier, auroc, bal_acc, bias, variance, len(cal_idx), len(sim_idx)),
            })
            print_full_metrics(predictor_name, brier, auroc, bal_acc, bias, variance)

            # Save c_hat arrays for simulation, preserving other predictors.
            label_dir = 'perturbed_labels' if args.perturbed_labels else 'standard_labels'
            out_dir = RESULTS_DIR / label_dir / benchmark
            out_dir.mkdir(parents=True, exist_ok=True)
            roberta_path = out_dir / f'c_hat_roberta_{model}{qonly_suffix}.npz'
            existing = {}
            if roberta_path.exists():
                with np.load(roberta_path) as data:
                    existing = {k: data[k] for k in data}
            existing[predictor_name] = c_hat_sim
            np.savez(roberta_path, **existing)

    # Save results.
    quality_df = pd.DataFrame(quality_rows)
    quality_df.to_csv(RESULTS_DIR / f'predictor_quality{qonly_suffix}.csv', index=False)
    quality_df.to_parquet(RESULTS_DIR / f'predictor_quality{qonly_suffix}.parquet')

    # Write markdown table.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    md = format_quality_table(quality_df)
    md_path = FIGURES_DIR / f'brier_table{qonly_suffix}.md'
    md_path.write_text(md)
    print(f'\n{md}')
    print(f'\nSaved: {md_path}')


if __name__ == '__main__':
    main()
