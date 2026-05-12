"""Shared metrics helpers used by multiple paper evaluation scripts."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from hubble.benchmarks import DOSE_GROUPS
from hubble.simulation import DIFFICULTY_BINS


def safe_auroc(y_true, y_score) -> float:
    """AUROC that returns NaN for single-class inputs."""
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_score)


def difficulty_bin_masks(confidence, duplicates):
    """Assign contaminated items to difficulty bins using confidence terciles."""
    contam_mask = duplicates > 0
    if contam_mask.sum() == 0 or confidence is None:
        return {b: np.zeros(len(duplicates), dtype=bool) for b in DIFFICULTY_BINS}

    conf = confidence[contam_mask]
    edges = np.quantile(conf, [1 / 3, 2 / 3])
    bins = np.digitize(conf, edges)

    bin_index = {'hard': 0, 'medium': 1, 'easy': 2}
    masks = {}
    contam_indices = np.where(contam_mask)[0]
    for name, idx in bin_index.items():
        mask = np.zeros(len(duplicates), dtype=bool)
        mask[contam_indices[bins == idx]] = True
        masks[name] = mask
    return masks


def metrics_by_dose(
    c_hat,
    y_clean,
    duplicates,
    *,
    confidence=None,
    include_balanced_accuracy: bool = False,
    include_variance: bool = False,
):
    """Compute shared paper metrics by dose group and optionally difficulty bin."""
    brier = {}
    auroc = {}
    bias = {}
    bal_acc = {}
    variance = {}
    c_pred = (c_hat >= 0.5).astype(int)

    for group_name, doses in DOSE_GROUPS.items():
        mask = (duplicates == 0) | np.isin(duplicates, doses) if group_name == 'low' else np.isin(duplicates, doses)
        if mask.sum() == 0:
            brier[group_name] = float('nan')
            auroc[group_name] = float('nan')
            bias[group_name] = float('nan')
            if include_balanced_accuracy:
                bal_acc[group_name] = float('nan')
            if include_variance:
                variance[group_name] = float('nan')
            continue

        brier[group_name] = brier_score_loss(y_clean[mask], c_hat[mask])
        auroc[group_name] = safe_auroc(y_clean[mask], c_hat[mask])
        bias[group_name] = np.abs(np.mean(c_hat[mask] - y_clean[mask]))
        if include_balanced_accuracy:
            bal_acc[group_name] = (
                float('nan')
                if len(np.unique(y_clean[mask])) < 2
                else balanced_accuracy_score(y_clean[mask], c_pred[mask])
            )
        if include_variance:
            variance[group_name] = np.var(c_hat[mask] - y_clean[mask])

    diff_masks = (
        difficulty_bin_masks(confidence, duplicates) if confidence is not None
        else {b: np.zeros(len(duplicates), dtype=bool) for b in DIFFICULTY_BINS}
    )
    for bin_name in DIFFICULTY_BINS:
        mask = diff_masks[bin_name]
        if mask.sum() == 0:
            brier[bin_name] = float('nan')
            auroc[bin_name] = float('nan')
            bias[bin_name] = float('nan')
            if include_balanced_accuracy:
                bal_acc[bin_name] = float('nan')
            if include_variance:
                variance[bin_name] = float('nan')
            continue

        brier[bin_name] = brier_score_loss(y_clean[mask], c_hat[mask])
        auroc[bin_name] = safe_auroc(y_clean[mask], c_hat[mask])
        bias[bin_name] = np.abs(np.mean(c_hat[mask] - y_clean[mask]))
        if include_balanced_accuracy:
            bal_acc[bin_name] = (
                float('nan')
                if len(np.unique(y_clean[mask])) < 2
                else balanced_accuracy_score(y_clean[mask], c_pred[mask])
            )
        if include_variance:
            variance[bin_name] = np.var(c_hat[mask] - y_clean[mask])

    diff_all_mask = diff_masks['easy'] | diff_masks['medium'] | diff_masks['hard']
    if diff_all_mask.sum() == 0:
        bias['diff_all'] = float('nan')
        if include_variance:
            variance['diff_all'] = float('nan')
    else:
        bias['diff_all'] = np.abs(np.mean(c_hat[diff_all_mask] - y_clean[diff_all_mask]))
        if include_variance:
            variance['diff_all'] = np.var(c_hat[diff_all_mask] - y_clean[diff_all_mask])

    brier['all'] = brier_score_loss(y_clean, c_hat)
    brier['clean'] = brier_score_loss(y_clean[duplicates == 0], c_hat[duplicates == 0])
    brier['contaminated'] = brier_score_loss(y_clean[duplicates > 0], c_hat[duplicates > 0])

    auroc['all'] = safe_auroc(y_clean, c_hat)
    auroc['clean'] = safe_auroc(y_clean[duplicates == 0], c_hat[duplicates == 0])
    auroc['contaminated'] = safe_auroc(y_clean[duplicates > 0], c_hat[duplicates > 0])

    bias['all'] = np.abs(np.mean(c_hat - y_clean))
    bias['clean'] = np.abs(np.mean(c_hat[duplicates == 0] - y_clean[duplicates == 0]))
    bias['contaminated'] = np.abs(np.mean(c_hat[duplicates > 0] - y_clean[duplicates > 0]))

    if include_balanced_accuracy:
        bal_acc['all'] = float('nan') if len(np.unique(y_clean)) < 2 else balanced_accuracy_score(y_clean, c_pred)
        bal_acc['clean'] = (
            float('nan')
            if len(np.unique(y_clean[duplicates == 0])) < 2
            else balanced_accuracy_score(y_clean[duplicates == 0], c_pred[duplicates == 0])
        )
        bal_acc['contaminated'] = (
            float('nan')
            if len(np.unique(y_clean[duplicates > 0])) < 2
            else balanced_accuracy_score(y_clean[duplicates > 0], c_pred[duplicates > 0])
        )

    if include_variance:
        variance['all'] = np.var(c_hat - y_clean)
        variance['clean'] = np.var(c_hat[duplicates == 0] - y_clean[duplicates == 0])
        variance['contaminated'] = np.var(c_hat[duplicates > 0] - y_clean[duplicates > 0])

    if include_balanced_accuracy and include_variance:
        return brier, auroc, bal_acc, bias, variance
    if include_balanced_accuracy:
        return brier, auroc, bal_acc, bias
    if include_variance:
        return brier, auroc, bias, variance
    return brier, auroc, bias


def quality_row(benchmark, model, probe_name, brier, auroc, bias, n_cal, n_sim, **extra):
    """Build a standard quality-results row used by several paper scripts."""
    row = {
        'benchmark': benchmark,
        'model': model,
        'probe': probe_name,
        'brier_clean': brier['clean'],
        'brier_contaminated': brier['contaminated'],
        'brier_all': brier['all'],
        'brier_low': brier['low'],
        'brier_mid': brier['mid'],
        'brier_high': brier['high'],
        'auroc_clean': auroc['clean'],
        'auroc_contaminated': auroc['contaminated'],
        'auroc_all': auroc['all'],
        'auroc_low': auroc['low'],
        'auroc_mid': auroc['mid'],
        'auroc_high': auroc['high'],
        'bias_clean': bias['clean'],
        'bias_contaminated': bias['contaminated'],
        'bias_all': bias['all'],
        'bias_low': bias['low'],
        'bias_mid': bias['mid'],
        'bias_high': bias['high'],
        'n_cal': n_cal,
        'n_sim': n_sim,
    }
    row.update(extra)
    return row


def print_metrics(probe_name, brier, auroc, bias):
    """Print a compact one-line summary for a probe."""
    print(
        f'  {probe_name:<20s}  '
        f'brier: clean={brier["clean"]:.4f} contam={brier["contaminated"]:.4f} all={brier["all"]:.4f}  '
        f'auroc: clean={auroc["clean"]:.4f} contam={auroc["contaminated"]:.4f} all={auroc["all"]:.4f}  '
        f'bias: low={bias["low"]:.4f} mid={bias["mid"]:.4f} high={bias["high"]:.4f} all={bias["all"]:.4f}'
    )


__all__ = [
    'safe_auroc',
    'difficulty_bin_masks',
    'metrics_by_dose',
    'quality_row',
    'print_metrics',
]
