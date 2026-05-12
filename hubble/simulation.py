"""Simulation framework for evaluating contamination correction estimators.

Benchmark-agnostic: takes per-item outcome arrays and runs Monte Carlo
simulations to compare estimator MSE under different contamination regimes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache


import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ItemPool:
    """Pool of items available for sampling into synthetic test sets.

    All arrays have length n_items (the total number of available items).
    Optional fields (confidence, d_hat, c_hat) are validated on assignment
    so that length mismatches are caught immediately.
    """

    y_observed: np.ndarray  # binary outcomes from perturbed model
    y_clean: np.ndarray     # binary outcomes from standard model (ground truth)
    duplicates: np.ndarray  # true duplication counts per item
    confidence: np.ndarray | None = None  # standard model confidence on correct answer
    d_hat: np.ndarray | None = None  # P(contaminated | x), from a memorization probe
    c_hat: np.ndarray | None = None  # E[Y_clean | x], from a correctness probe

    def __post_init__(self):
        n = len(self.y_observed)
        for arr in (self.y_clean, self.duplicates, self.confidence, self.d_hat, self.c_hat):
            if arr is not None:
                assert len(arr) == n

    @property
    def n_items(self) -> int:
        return len(self.y_observed)

    @property
    def clean_idx(self) -> np.ndarray:
        """Indices of items with zero duplication (truly clean)."""
        return np.where(self.duplicates == 0)[0]

    @property
    def contaminated_idx(self) -> np.ndarray:
        """Indices of items with nonzero duplication."""
        return np.where(self.duplicates > 0)[0]

    def items_by_dose(self, doses: list[int]) -> np.ndarray:
        """Indices of contaminated items at specific duplication levels."""
        mask = np.isin(self.duplicates, doses)
        return np.where(mask)[0]

    @classmethod
    def from_eval_parquets(
        cls,
        std_path: str,
        prt_path: str,
        acc_clean_col: str,
        acc_perturbed_col: str,
        confidence_col: str | None = None,
        duplicates_col: str = "duplicates",
    ) -> "ItemPool":
        """Construct an ItemPool from standard and perturbed evaluation parquets.

        Builds the hybrid observed outcome: standard model on clean items,
        perturbed model on contaminated items. This isolates contamination as
        the sole source of observed–clean discrepancy.

        Args:
            std_path:          Path to standard-model eval parquet.
            prt_path:          Path to perturbed-model eval parquet.
            acc_clean_col:     Column name for clean accuracy in std_path.
            acc_perturbed_col: Column name for perturbed accuracy in prt_path.
            confidence_col:    Column name for confidence in std_path.
            duplicates_col:    Column name for duplication counts (default "duplicates").
        """
        std = pd.read_parquet(std_path)
        prt = pd.read_parquet(prt_path)

        assert len(std) == len(prt), (
            f'Row count mismatch: std has {len(std)} rows, prt has {len(prt)} rows'
        )
        std[acc_perturbed_col] = prt[acc_perturbed_col].values

        duplicates = std[duplicates_col].values.astype(int)
        y_clean = std[acc_clean_col].values.astype(int)
        y_observed = np.where(
            duplicates > 0, std[acc_perturbed_col].values, y_clean).astype(int)
        confidence = std[confidence_col].values if confidence_col else None

        return cls(y_observed=y_observed, y_clean=y_clean, duplicates=duplicates,
                   confidence=confidence)


def stratified_split(
    pool: ItemPool, seed: int = 42,
) -> tuple[ItemPool, ItemPool, np.ndarray, np.ndarray]:
    """Split an ItemPool into two halves, stratified by duplication level.

    Deterministic for a given seed, so calibration and simulation pools
    are always the same across experiments.

    Returns (calibration_pool, simulation_pool, cal_idx, sim_idx).
    """
    rng = np.random.default_rng(seed)
    cal_idx, sim_idx = [], []
    for dup_level in sorted(np.unique(pool.duplicates)):
        level_idx = np.where(pool.duplicates == dup_level)[0]
        rng.shuffle(level_idx)
        half = len(level_idx) // 2
        cal_idx.append(level_idx[:half])
        sim_idx.append(level_idx[half:])
    cal_idx = np.concatenate(cal_idx)
    sim_idx = np.concatenate(sim_idx)

    def _subset(idx):
        return ItemPool(
            y_observed=pool.y_observed[idx],
            y_clean=pool.y_clean[idx],
            duplicates=pool.duplicates[idx],
            confidence=pool.confidence[idx] if pool.confidence is not None else None,
            d_hat=pool.d_hat[idx] if pool.d_hat is not None else None,
            c_hat=pool.c_hat[idx] if pool.c_hat is not None else None,
        )

    return _subset(cal_idx), _subset(sim_idx), cal_idx, sim_idx


@dataclass
class TestSet:
    """A sampled test set with probe predictions attached."""

    y_observed: np.ndarray  # (n,) binary outcomes from perturbed model
    y_clean: np.ndarray     # (n,) binary outcomes from standard model
    d_hat: np.ndarray       # (n,) P(contaminated | x), continuous in [0, 1]
    c_hat: np.ndarray       # (n,) E[Y_clean | x], continuous in [0, 1]
    indices: np.ndarray     # (n,) indices into the original ItemPool

    @property
    def ground_truth(self) -> float:
        """True clean accuracy for this test set."""
        return self.y_clean.mean()


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------

def naive_estimator(ts: TestSet) -> float:
    """No correction: just average observed outcomes."""
    return ts.y_observed.mean()


def ipw_estimator(ts: TestSet, eps: float = 1e-8) -> float:
    """Soft IPW: downweight items proportional to P(contaminated)."""
    w = 1 - ts.d_hat
    if w.sum() < eps:
        return np.nan
    return np.average(ts.y_observed, weights=w)


def ipw_hard_estimator(ts: TestSet, eps: float = 1e-8) -> float:
    """Hard IPW: downweight items proportional to P(contaminated).
    
    Note: Requires the d_hat scores to be calibrated and lie in [0, 1].
    """
    w = (1 - ts.d_hat) >= 0.5  # threshold at 0.5, i.e. route items with P(contaminated) >= 0.5
    if w.sum() < eps:
        return np.nan
    return np.average(ts.y_observed, weights=w)


def epg_ipw_estimator(ts: TestSet, n_thresholds: int = 200) -> float:
    """EPG-IPW: ConTAM-inspired threshold selection on d_hat.

    Sweeps thresholds over d_hat (P(contaminated | x)), computes the z-score
    at each threshold (EPG / SE), selects the threshold that maximizes z-score,
    and returns the accuracy on the "clean" subset (d_hat <= threshold).

    Note: This adds an explicit criterion to avoid aggressive thresholds that 
    mark a large number of items as contaminated.
    
    Note: Requires the d_hat scores to lie in [0, 1] but doesn't require calibration.
    """
    d_hat = ts.d_hat
    y_obs = ts.y_observed
    acc_full = y_obs.mean()
    sigma = np.std(y_obs)

    thresholds = np.linspace(min(1.0, d_hat.max()), max(0.0, d_hat.min()), n_thresholds)

    best_z = 0.0  # For no correction (t=1.0), EPG=0 => z=0, so this is the baseline to beat
    best_t = 1.0  # For debugging
    best_acc_clean = acc_full  # fallback: no correction

    for t in thresholds:
        clean_mask = d_hat <= t
        n_clean = clean_mask.sum()

        if n_clean == 0 or clean_mask.mean() <0.1:  # require at least 10% of items to be "clean" for stability
            continue

        acc_clean = y_obs[clean_mask].mean()
        epg = acc_full - acc_clean
        se = sigma / np.sqrt(n_clean)
        z = epg / se if se > 0 else 0.0

        if z > best_z:
            best_z = z
            best_acc_clean = acc_clean
            best_t = t
    
    return best_acc_clean


def imputation_estimator(ts: TestSet) -> float:
    """Replace all outcomes with correctness probe predictions."""
    return ts.c_hat.mean()


def combined_estimator(ts: TestSet) -> float:
    """Route: use imputed value if flagged, observed if not."""
    return (ts.d_hat * ts.c_hat + (1 - ts.d_hat) * ts.y_observed).mean()


ESTIMATORS: dict[str, Callable] = {
    "naive": naive_estimator,
    "ipw": ipw_estimator,
    "imputation": imputation_estimator,
    "combined": combined_estimator,
}


# ---------------------------------------------------------------------------
# Synthetic probe generators
# ---------------------------------------------------------------------------

def _beta_auroc(mean_pos: float, concentration: float,
                n_mc: int = 100_000) -> float:
    """Monte Carlo AUROC for symmetric Beta probe.

    Positive ~ Beta(m*c, (1-m)*c), Negative ~ Beta((1-m)*c, m*c).
    Uses Mann-Whitney U via ranking for O(n log n) efficiency.
    """
    EPS = 1e-6
    a_pos = max(mean_pos * concentration, EPS)
    b_pos = max((1 - mean_pos) * concentration, EPS)
    rng = np.random.default_rng(0)  # fixed seed for deterministic bisection
    pos = rng.beta(a_pos, b_pos, size=n_mc)
    neg = rng.beta(b_pos, a_pos, size=n_mc)
    scores = np.concatenate([pos, neg])
    order = np.argsort(scores)
    ranks = np.empty(2 * n_mc, dtype=float)
    ranks[order] = np.arange(1, 2 * n_mc + 1, dtype=float)
    pos_rank_sum = ranks[:n_mc].sum()
    return (pos_rank_sum - n_mc * (n_mc + 1) / 2) / (n_mc * float(n_mc))


@lru_cache(maxsize=128)
def _find_mean_for_auroc(target_auroc: float, concentration: float,
                         n_mc: int = 100_000) -> float:
    """Bisect to find mean_pos in [0.5, 1) that yields target AUROC."""
    if target_auroc <= 0.5:
        return 0.5
    lo, hi = 0.5, 1.0 - 1e-9
    for _ in range(60):
        mid = (lo + hi) / 2
        if _beta_auroc(mid, concentration, n_mc) < target_auroc:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-8:
            break
    return (lo + hi) / 2


def make_synthetic_memorization_probe(
    true_labels: np.ndarray,
    alpha: float,
    rng: np.random.Generator,
    concentration: float = 10.0,
) -> np.ndarray:
    """Generate a synthetic memorization probe that outputs P(contaminated | x).

    Simulates a continuous-valued contamination detector by drawing scores from
    Beta distributions whose means are separated to achieve a target AUROC.
    For contaminated items (true_labels=1), scores are drawn from Beta(mean_pos)
    skewed high; for clean items (true_labels=0), scores are drawn from
    Beta(1-mean_pos) skewed low. The mean_pos parameter is found via bisection
    to match the requested AUROC.

    Args:
        true_labels: Binary array indicating contaminated (1) vs clean (0) items.
        alpha: Target AUROC of the probe. 0.5 = random, 1.0 = perfect separation.
        rng: NumPy random generator for reproducibility.
        concentration: Controls the tightness (inverse variance) of the Beta
            distributions. Higher values produce less noisy scores.

    Returns:
        Array of probe scores in [0, 1], one per item.
    """
    mean_pos = _find_mean_for_auroc(alpha, concentration)
    EPS = 1e-6

    # Per-item Beta mean: high for contaminated, low for clean
    mean = np.empty_like(true_labels, dtype=float)
    mean[true_labels == 1] = mean_pos
    mean[true_labels == 0] = 1 - mean_pos

    # Beta shape parameters, floored to avoid degenerate distributions
    a = np.maximum(mean * concentration, EPS)
    b = np.maximum((1 - mean) * concentration, EPS)
    return rng.beta(a, b)


def make_synthetic_correctness_probe(
    y_clean: np.ndarray,
    d: float,
    rng: np.random.Generator,
    concentration: float = 10.0,
    base_rate: float = 0.5,
) -> np.ndarray:
    """Generate a synthetic correctness probe that outputs E[Y_clean | x].

    Simulates a continuous-valued predictor of clean accuracy by drawing scores
    from Beta distributions. Items answered correctly (y_clean=1) get scores
    drawn from a Beta skewed toward 1; incorrect items (y_clean=0) get scores
    skewed toward 0.

    The informativeness parameter d controls how far per-class means shift
    from base_rate toward the truth. The imputation estimator's bias on a set
    of items with clean accuracy p_S is: bias = (1 - d) * (base_rate - p_S).

    Args:
        y_clean: Binary array of clean outcomes (1 = correct, 0 = incorrect).
        d: Informativeness in [0, 1]. d=0 means uninformative (always predicts
            base_rate), d=1 means perfect (predicts 1 for correct, 0 for incorrect).
        rng: NumPy random generator for reproducibility.
        concentration: Controls the tightness (inverse variance) of the Beta
            distributions. Does not affect expected bias, only score variance.
        base_rate: Chance-level accuracy for the task (e.g. 0.5 for binary,
            0.25 for 4-way MCQ). Used as the uninformative prediction baseline.

    Returns:
        Array of probe scores in [0, 1], one per item.
    """
    # Per-class Beta means: shift from base_rate toward the truth, proportional to d.
    # d=0 => both collapse to base_rate; d=1 => mean_correct=1, mean_incorrect=0.
    mean_correct = base_rate + d * (1 - base_rate)
    mean_incorrect = base_rate - d * base_rate

    # Sample from Beta(a, b) per item. Concentration controls noise around the
    # class mean but does not affect expected MAE (which depends only on means).
    EPS = 1e-6
    mean = np.empty_like(y_clean, dtype=float)
    mean[y_clean == 1] = mean_correct
    mean[y_clean == 0] = mean_incorrect
    a = np.maximum(mean * concentration, EPS)
    b = np.maximum((1 - mean) * concentration, EPS)
    return rng.beta(a, b)


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

DOSE_GROUPS = {
    "all": [1, 4, 16, 64, 256],
    "high": [64, 256],
    "low": [1],
    "mid": [16],
}


DIFFICULTY_BINS = ("easy", "medium", "hard")


@dataclass
class SamplerConfig:
    """Configuration for how test sets are sampled from the item pool."""

    regime: str = "random"            # "random" or "correlated"
    n: int = 500                      # test set size
    gamma: float = 0.3                # contamination rate
    dose_group: str = "mid"           # which duplication levels to include
    difficulty_bin: str = "hard"      # which tercile to draw contaminated items from
                                      #   (correlated only): "hard", "medium", or "easy"


def sample_test_set(
    pool: ItemPool,
    config: SamplerConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample item indices from pool according to config."""
    if config.regime == "random":
        return sample_random(pool, config.n, config.gamma, config.dose_group, rng)
    elif config.regime == "correlated":
        return sample_correlated(
            pool, config.n, config.gamma, config.difficulty_bin, config.dose_group, rng
        )
    else:
        raise ValueError(f"Unknown regime: {config.regime}")


def sample_random(
    pool: ItemPool,
    n: int,
    gamma: float,
    dose_group: str = "all",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a test set under random contamination.

    Draws n*(1-gamma) items from the clean pool and n*gamma from the
    contaminated pool (filtered by dose_group). Returns item indices.
    """
    if rng is None:
        rng = np.random.default_rng()

    doses = DOSE_GROUPS[dose_group]
    clean_pool = pool.clean_idx
    contam_pool = pool.items_by_dose(doses)

    n_contam = int(round(n * gamma))
    n_clean = n - n_contam

    clean_sample = rng.choice(clean_pool, size=n_clean, replace=True)
    contam_sample = rng.choice(contam_pool, size=n_contam, replace=True)
    return np.concatenate([clean_sample, contam_sample])


def sample_correlated(
    pool: ItemPool,
    n: int,
    gamma: float,
    difficulty_bin: str = "hard",
    dose_group: str = "all",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample a test set where contamination is correlated with difficulty.

    Bins contaminated items into terciles by confidence, then draws
    contaminated items only from the specified bin.
      "hard":   lowest confidence tercile
      "medium": middle confidence tercile
      "easy":   highest confidence tercile
    """
    assert pool.confidence is not None, (
        'Correlated sampling requires confidence scores on the ItemPool')

    if rng is None:
        rng = np.random.default_rng()

    doses = DOSE_GROUPS[dose_group]
    contam_pool = pool.items_by_dose(doses)
    clean_pool = pool.clean_idx

    # NOTE: [pedagogical] np.digitize assigns each value to a bin defined by
    # edges. With tercile edges, bin 0 = hard (low confidence), 1 = medium, 2 = easy.
    bin_index = {'hard': 0, 'medium': 1, 'easy': 2}[difficulty_bin]
    conf = pool.confidence[contam_pool]
    bins = np.digitize(conf, np.quantile(conf, [1/3, 2/3]))
    bin_pool = contam_pool[bins == bin_index]

    n_contam = int(round(n * gamma))
    n_clean = n - n_contam

    contam_sample = rng.choice(bin_pool, size=n_contam, replace=True)
    clean_sample = rng.choice(clean_pool, size=n_clean, replace=True)
    return np.concatenate([clean_sample, contam_sample])


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------

def run_simulation(
    pool: ItemPool,
    sampler_config: SamplerConfig,
    *,
    n_replicates: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run simulation: sample test sets, apply estimators, compare to ground truth.

    Requires pool.d_hat and pool.c_hat to be set. Use
    make_synthetic_memorization_probe / make_synthetic_correctness_probe
    to generate synthetic probes, or attach real probe predictions.

    Returns DataFrame with columns:
        replicate, estimator, estimate, ground_truth, error, squared_error
    """
    assert pool.d_hat is not None, 'pool.d_hat must be set before running simulation'
    assert pool.c_hat is not None, 'pool.c_hat must be set before running simulation'

    rng = np.random.default_rng(seed)
    records = []
    for rep in range(n_replicates):
        indices = sample_test_set(pool, sampler_config, rng)

        ts = TestSet(
            y_observed=pool.y_observed[indices],
            y_clean=pool.y_clean[indices],
            d_hat=pool.d_hat[indices],
            c_hat=pool.c_hat[indices],
            indices=indices,
        )
        gt = ts.ground_truth

        for name, est_fn in ESTIMATORS.items():
            est = est_fn(ts)
            records.append({
                "replicate": rep,
                "estimator": name,
                "estimate": est,
                "ground_truth": gt,
                "error": est - gt,
                "squared_error": (est - gt) ** 2,
            })

    return pd.DataFrame(records)


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-replicate results to bias, variance, MSE per estimator."""
    return df.groupby("estimator").agg(
        bias=("error", "mean"),
        variance=("error", lambda x: x.var()),
        mse=("squared_error", "mean"),
        mean_estimate=("estimate", "mean"),
        mean_ground_truth=("ground_truth", "mean"),
        n_replicates=("replicate", "nunique"),
    ).reset_index()
