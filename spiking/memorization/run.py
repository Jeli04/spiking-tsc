"""Generate memorization predictor d_hat predictions for all benchmarks.

Fits MIA predictors (Platt-scaled), hidden state predictors, and residual stream
predictors (logistic regression) on the calibration split, predicts d_hat
(P(contaminated)) on the simulation split. Output d_hat arrays are compatible
with sim_pool.d_hat for run_simulation.

MIA scores, hidden-state features, and residual features are loaded from cache.
No GPU needed.

Usage:
  uv run python src/spiking/memorization/run.py
  uv run python src/spiking/memorization/run.py --benchmark popqa
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from spiking.config import ATTACKS, BENCHMARK_EXP11_MAP, DOSE_GROUPS, MODELS
from hubble.mem_predictors import MIAPredictor, ResidualPredictor, RESIDUAL_PREDICTORS
from hubble.predictors import PREDICTORS
from hubble.simulation import ItemPool, stratified_split

RESULTS_DIR = Path(__file__).parent / 'results'
FIGURES_DIR = Path(__file__).parent / 'figures'
PAPER_DIR = Path(__file__).resolve().parent.parent  # src/spiking/
DATA_RESULTS = PAPER_DIR / 'data_generation' / 'results'
EXP50_DIR = DATA_RESULTS
EXP11_SCORES = DATA_RESULTS / 'all_scores.parquet'
EXP12_FEATURES = DATA_RESULTS / 'features'
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESIDUAL_CACHE_DIR = PROJECT_ROOT / 'data' / 'hubble-8b-500b_toks-perturbed-hf-residual-cache'

# Helpers

def load_item_pool(benchmark, model):
    """Load the full ItemPool from release eval parquets.

    PopQA uses exact_match_ instead of acc_, and Wikipedia uses continuous loss.
    Wikipedia only needs duplicate labels here, so it gets dummy outcomes.
    """
    tag = f'hubble-{model}_toks'
    std_path = EXP50_DIR / benchmark / f'eval_{tag}-standard-hf.parquet'
    prt_path = EXP50_DIR / benchmark / f'eval_{tag}-perturbed-hf.parquet'

    std_df = pd.read_parquet(std_path)

    # Wikipedia has loss instead of binary accuracy.
    if f'loss_{tag}-standard-hf' in std_df.columns:
        n = len(std_df)
        duplicates = std_df['duplicates'].values.astype(int)
        return ItemPool(
            y_observed=np.zeros(n, dtype=int),
            y_clean=np.zeros(n, dtype=int),
            duplicates=duplicates,
        )

    # PopQA uses exact_match_; MCQ benchmarks use acc_.
    if f'acc_{tag}-standard-hf' in std_df.columns:
        acc_prefix = 'acc'
    else:
        acc_prefix = 'exact_match'

    confidence_col = f'confidence_{tag}-standard-hf'
    has_confidence = confidence_col in std_df.columns

    return ItemPool.from_eval_parquets(
        str(std_path), str(prt_path),
        acc_clean_col=f'{acc_prefix}_{tag}-standard-hf',
        acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
        confidence_col=confidence_col if has_confidence else None,
    )


def load_mia_scores(exp11_benchmark, model, exp11_format=None):
    """Load cached MIA attack scores."""
    all_scores = pd.read_parquet(EXP11_SCORES)
    mask = (all_scores['benchmark'] == exp11_benchmark) & (all_scores['model'] == model)
    if exp11_format is not None:
        mask = mask & (all_scores['format'] == exp11_format)
    df = all_scores[mask].reset_index(drop=True)
    return df[ATTACKS].values, ATTACKS


def load_hidden_features(exp12_benchmark, model, pool_name, exp12_format=None):
    """Load cached hidden-state features."""
    label = f'hubble-{model}_toks-perturbed-hf'
    feat_path = EXP12_FEATURES / exp12_benchmark / f'features_{label}_{pool_name}.npz'
    meta_path = EXP12_FEATURES / exp12_benchmark / f'meta_{label}.parquet'

    features = np.load(feat_path)['hidden_states']
    if exp12_format is not None:
        meta = pd.read_parquet(meta_path)
        mask = (meta['format'] == exp12_format).values
        features = features[mask]
    return features


def load_residual_features(benchmark, layer, pool_name, exp_format=None):
    """Load cached residual-stream features."""
    feat_path = RESIDUAL_CACHE_DIR / benchmark / f'features_layer{layer:02d}_{pool_name}.npz'
    meta_path = RESIDUAL_CACHE_DIR / benchmark / 'meta.parquet'

    features = np.load(feat_path)['hidden_states']
    if exp_format is not None:
        meta = pd.read_parquet(meta_path)
        mask = (meta['format'] == exp_format).values
        features = features[mask]
    return features


def auroc_by_dose(d_hat, duplicates):
    """Compute AUROC for each dose group (vs clean items).

    Returns dict mapping dose group name to AUROC. Each evaluation includes
    only clean (dup=0) items and items at the specified dose levels.
    """
    clean_mask = duplicates == 0
    results = {}
    for group_name, doses in DOSE_GROUPS.items():
        dose_mask = np.isin(duplicates, doses)
        mask = clean_mask | dose_mask
        if mask.sum() == 0 or dose_mask.sum() == 0:
            results[group_name] = float('nan')
            continue
        labels = (duplicates[mask] > 0).astype(int)
        try:
            results[group_name] = roc_auc_score(labels, d_hat[mask])
        except ValueError:
            results[group_name] = float('nan')
    return results


def format_auroc_table(quality_df):
    """Format AUROC results as a markdown table grouped by benchmark/model."""
    lines = []
    for (benchmark, model), group in quality_df.groupby(['benchmark', 'model']):
        lines.append(f'\n### {benchmark} / {model}\n')
        lines.append('| Attack | Low (1x) | Mid (4-16x) | High (64-256x) | All |')
        lines.append('|--------|----------|-------------|----------------|-----|')
        for _, row in group.iterrows():
            lines.append(
                f'| {row["attack"]:<20s} '
                f'| {row["auroc_low"]:.3f}    '
                f'| {row["auroc_mid"]:.3f}       '
                f'| {row["auroc_high"]:.3f}          '
                f'| {row["auroc_all"]:.3f} |'
            )
    return '\n'.join(lines)


# Main

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default=None,
                        choices=list(BENCHMARK_EXP11_MAP.keys()),
                        help='Single benchmark (default: all)')
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    quality_rows = []

    if args.benchmark:
        benchmarks_to_run = {args.benchmark: BENCHMARK_EXP11_MAP[args.benchmark]}
    else:
        benchmarks_to_run = BENCHMARK_EXP11_MAP

    for benchmark, mapping in benchmarks_to_run.items():
        for model in MODELS:
            print(f'\n=== {benchmark} / {model} ===')

            pool = load_item_pool(benchmark, model)
            cal_pool, sim_pool, cal_idx, sim_idx = stratified_split(pool)

            scores_matrix, attack_names = load_mia_scores(
                mapping['exp11_benchmark'], model, mapping['exp11_format'])
            if len(scores_matrix) == 0:
                print('  MIA scores not available, skipping MIA predictors')
                has_mia = False
            else:
                assert len(scores_matrix) == pool.n_items, (
                    f'Row mismatch: {len(scores_matrix)} scores vs {pool.n_items} items')
                has_mia = True

            labels_cal = (cal_pool.duplicates > 0).astype(int)
            labels_sim = (sim_pool.duplicates > 0).astype(int)

            d_hat_dict = {}

            # MIA predictors.
            for i, attack in enumerate(attack_names if has_mia else []):
                scores = scores_matrix[:, i]
                predictor = MIAPredictor(attack)
                predictor.fit(scores[cal_idx], labels_cal)
                d_hat_sim = predictor.predict_proba(scores[sim_idx])
                d_hat_dict[attack] = d_hat_sim

                dose_aurocs = auroc_by_dose(d_hat_sim, sim_pool.duplicates)
                try:
                    auroc_all = roc_auc_score(labels_sim, d_hat_sim)
                except ValueError:
                    auroc_all = float('nan')

                quality_rows.append({
                    'benchmark': benchmark, 'model': model, 'attack': attack,
                    'auroc_low': dose_aurocs['low'],
                    'auroc_mid': dose_aurocs['mid'],
                    'auroc_high': dose_aurocs['high'],
                    'auroc_all': auroc_all,
                    'n_cal': len(cal_idx), 'n_sim': len(sim_idx),
                })
                print(f'  {attack:<20s}  '
                      f'low={dose_aurocs["low"]:.3f}  '
                      f'mid={dose_aurocs["mid"]:.3f}  '
                      f'high={dose_aurocs["high"]:.3f}  '
                      f'all={auroc_all:.3f}')

            # Hidden-state predictors.
            for predictor_name, predictor_cls in PREDICTORS.items():
                pool_name = predictor_cls.pool.__name__
                try:
                    features = load_hidden_features(
                        mapping['exp11_benchmark'], model,
                        pool_name, mapping['exp11_format'])
                except FileNotFoundError:
                    print(f'  {predictor_name:<20s}  SKIPPED (no cached features)')
                    continue

                assert len(features) == pool.n_items, (
                    f'Row mismatch: {len(features)} features vs {pool.n_items} items')

                predictor = predictor_cls()
                predictor.fit(features[cal_idx], labels_cal)
                d_hat_sim = predictor.predict_proba(features[sim_idx])
                d_hat_dict[predictor_name] = d_hat_sim

                dose_aurocs = auroc_by_dose(d_hat_sim, sim_pool.duplicates)
                try:
                    auroc_all = roc_auc_score(labels_sim, d_hat_sim)
                except ValueError:
                    auroc_all = float('nan')

                quality_rows.append({
                    'benchmark': benchmark, 'model': model, 'attack': predictor_name,
                    'auroc_low': dose_aurocs['low'],
                    'auroc_mid': dose_aurocs['mid'],
                    'auroc_high': dose_aurocs['high'],
                    'auroc_all': auroc_all,
                    'n_cal': len(cal_idx), 'n_sim': len(sim_idx),
                })
                print(f'  {predictor_name:<20s}  '
                      f'low={dose_aurocs["low"]:.3f}  '
                      f'mid={dose_aurocs["mid"]:.3f}  '
                      f'high={dose_aurocs["high"]:.3f}  '
                      f'all={auroc_all:.3f}')

            # Residual-stream predictors.
            for predictor_name, (layer, pool_name) in RESIDUAL_PREDICTORS.items():
                try:
                    features = load_residual_features(
                        mapping['exp11_benchmark'], layer,
                        pool_name, mapping['exp11_format'])
                except FileNotFoundError:
                    print(f'  {predictor_name:<20s}  SKIPPED (no cached features)')
                    continue

                assert len(features) == pool.n_items, (
                    f'Row mismatch: {len(features)} features vs {pool.n_items} items')

                predictor = ResidualPredictor(layer=layer, pool_name=pool_name)
                predictor.fit(features[cal_idx], labels_cal)
                d_hat_sim = predictor.predict_proba(features[sim_idx])
                d_hat_dict[predictor_name] = d_hat_sim

                dose_aurocs = auroc_by_dose(d_hat_sim, sim_pool.duplicates)
                try:
                    auroc_all = roc_auc_score(labels_sim, d_hat_sim)
                except ValueError:
                    auroc_all = float('nan')

                quality_rows.append({
                    'benchmark': benchmark, 'model': model, 'attack': predictor_name,
                    'auroc_low': dose_aurocs['low'],
                    'auroc_mid': dose_aurocs['mid'],
                    'auroc_high': dose_aurocs['high'],
                    'auroc_all': auroc_all,
                    'n_cal': len(cal_idx), 'n_sim': len(sim_idx),
                })
                print(f'  {predictor_name:<20s}  '
                      f'low={dose_aurocs["low"]:.3f}  '
                      f'mid={dose_aurocs["mid"]:.3f}  '
                      f'high={dose_aurocs["high"]:.3f}  '
                      f'all={auroc_all:.3f}')

            # Save d_hat arrays for simulation.
            out_dir = RESULTS_DIR / benchmark
            out_dir.mkdir(parents=True, exist_ok=True)
            np.savez(out_dir / f'd_hat_{model}.npz', **d_hat_dict)

    # Merge with existing results for benchmarks not re-run.
    new_df = pd.DataFrame(quality_rows)
    csv_path = RESULTS_DIR / 'predictor_quality.csv'
    parquet_path = RESULTS_DIR / 'predictor_quality.parquet'

    if parquet_path.exists():
        existing_df = pd.read_parquet(parquet_path)
        # Replace rows for benchmarks we just re-ran.
        run_benchmarks = set(new_df['benchmark'].unique())
        kept_df = existing_df[~existing_df['benchmark'].isin(run_benchmarks)]
        quality_df = pd.concat([kept_df, new_df], ignore_index=True)
    else:
        quality_df = new_df

    quality_df.to_csv(csv_path, index=False)
    quality_df.to_parquet(parquet_path)

    # Write markdown table.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    md = format_auroc_table(quality_df)
    md_path = FIGURES_DIR / 'auroc_table.md'
    md_path.write_text(md)
    print(f'\n{md}')
    print(f'\nSaved: {md_path}')


if __name__ == '__main__':
    main()
