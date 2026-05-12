"""Simulation framework evaluation (Hubble controlled contamination).

Sweep memorization probe AUROC x correctness probe imputation bias under
random and correlated contamination at fixed gamma=0.3.

The correctness probe is parameterized by imputation bias (not MAE), since
bias is what matters for the imputation estimator. The key conversion is:
    bias = (1 - d) * |base_rate - p_test|
    d    = 1 - bias / |base_rate - p_test|
where d is the probe informativeness and p_test is the expected clean accuracy
on test sets for a given sampling regime. Since p_test differs across regimes,
the same bias value maps to different d values per panel.

No GPU needed.

Usage:
  uv run python src/spiking/simulation/run.py
  uv run python src/spiking/simulation/run.py --benchmark mmlu --n-replicates 100
"""

from hubble.simulation import (
    DOSE_GROUPS,
    ESTIMATORS,
    ItemPool,
    SamplerConfig,
    make_synthetic_correctness_probe,
    make_synthetic_memorization_probe,
    run_simulation,
    stratified_split,
    summarize_results,
)
import argparse
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from spiking.config import BENCHMARK_BASE_RATES, TEXT_WIDTH

# Match the paper template with LaTeX rendering.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{mathpazo}",
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
})

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
PAPER_DIR = Path(__file__).resolve().parent.parent  # src/spiking/
EXP50_DIR = PAPER_DIR / "data_generation" / "results"


# Bias <-> d conversion

def estimate_p_test(pool: ItemPool, gamma: float, dose_group: str,
                    regime: str, difficulty_bin: str = "hard") -> float:
    """Expected clean accuracy on test sets for a given sampling config.

    E[p_test] = (1 - gamma) * p_clean + gamma * p_contam_subset
    """
    p_clean = pool.y_clean[pool.clean_idx].mean()
    contam_idx = pool.items_by_dose(DOSE_GROUPS[dose_group])

    if regime == "correlated" and pool.confidence is not None:
        bin_index = {'hard': 0, 'medium': 1, 'easy': 2}[difficulty_bin]
        conf = pool.confidence[contam_idx]
        bins = np.digitize(conf, np.quantile(conf, [1/3, 2/3]))
        contam_idx = contam_idx[bins == bin_index]

    p_contam = pool.y_clean[contam_idx].mean()
    return (1 - gamma) * p_clean + gamma * p_contam


def bias_to_d(bias: float, base_rate: float, p_test: float) -> float:
    """Convert imputation bias to probe informativeness d.

    bias = (1 - d) * |base_rate - p_test|  =>  d = 1 - bias / |base_rate - p_test|
    """
    max_bias = abs(base_rate - p_test)
    if max_bias < 1e-10:
        return 1.0
    return np.clip(1 - bias / max_bias, 0.0, 1.0)


# Sweep

def sweep_probe_accuracy(
    pool: ItemPool,
    alphas_m: np.ndarray,
    biases_c: np.ndarray,
    p_test: float,
    gamma: float,
    n: int,
    n_replicates: int,
    seed: int,
    base_rate: float = 0.5,
    dose_group: str = "high",
    regime: str = "random",
    difficulty_bin: str = "hard",
) -> pd.DataFrame:
    """Sweep alpha_m (AUROC) x bias_c (imputation bias) under specified regime."""
    all_results = []
    total = len(alphas_m) * len(biases_c)
    i = 0

    for alpha_m, bias_c in product(alphas_m, biases_c):
        i += 1
        if i % 20 == 0 or i == total:
            print(f"  {i}/{total} (αm={alpha_m:.2f}, bias={bias_c:.3f})")

        # Convert bias to probe informativeness for this panel.
        d_c = bias_to_d(bias_c, base_rate, p_test)

        rng = np.random.default_rng(seed)
        true_contam = (pool.duplicates > 0).astype(int)
        pool.d_hat = make_synthetic_memorization_probe(
            true_contam, alpha_m, rng)
        pool.c_hat = make_synthetic_correctness_probe(
            pool.y_clean, d_c, rng, base_rate=base_rate)

        cfg = SamplerConfig(
            regime=regime, n=n, gamma=gamma,
            dose_group=dose_group, difficulty_bin=difficulty_bin,
        )
        df = run_simulation(pool, cfg, n_replicates=n_replicates, seed=seed)
        summary = summarize_results(df)
        summary["alpha_m"] = alpha_m
        summary["bias_c"] = bias_c
        all_results.append(summary)

    return pd.concat(all_results, ignore_index=True)


# Plotting

def plot_phase_panel(ax, df: pd.DataFrame, est_names: list, cmap, title: str):
    """Plot a single phase diagram panel onto the given axes."""
    best = df.loc[df.groupby(["alpha_m", "bias_c"])["mse"].idxmin()]
    pivot = best.pivot(index="bias_c", columns="alpha_m", values="estimator")

    est_to_int = {name: i for i, name in enumerate(est_names)}
    pivot_int = pivot.map(lambda x: est_to_int.get(x, -1)).astype(float)

    ax.imshow(
        pivot_int.values, cmap=cmap,
        vmin=-0.5, vmax=len(est_names) - 0.5, aspect="auto",
        extent=[
            pivot.columns.min(), pivot.columns.max(),
            pivot.index.max(), pivot.index.min(),
        ],
    )
    ax.set_title(title)


def plot_combined_phase(
    random_results: dict[str, pd.DataFrame],
    correlated_results: dict[str, pd.DataFrame],
    save_path: Path,
):
    """Phase diagrams: row 1 = random (by dose), row 2 = correlated (by difficulty) if available."""
    est_names = list(ESTIMATORS.keys())
    cmap = plt.colormaps.get_cmap("Set2").resampled(len(est_names))

    n_rows = 2 if correlated_results else 1
    height = TEXT_WIDTH * 0.6 if n_rows == 2 else TEXT_WIDTH * 0.35
    fig = plt.figure(figsize=(TEXT_WIDTH, height))
    gs = fig.add_gridspec(n_rows, 4, width_ratios=[1, 1, 1, 0.06],
                          wspace=0.12, hspace=0.20)

    axes = np.empty((n_rows, 3), dtype=object)
    row_data = [random_results]
    row_labels = ["Random"]
    row_titles = [[dg.capitalize() for dg in random_results]]
    if correlated_results:
        row_data.append(correlated_results)
        row_labels.append("Correlated")
        row_titles.append([k.capitalize() for k in correlated_results])

    for r, (data, titles) in enumerate(zip(row_data, row_titles)):
        for c, ((key, df), title) in enumerate(zip(data.items(), titles)):
            share_kw = {}
            if c > 0:
                share_kw = dict(sharex=axes[r, 0], sharey=axes[r, 0])
            elif r > 0:
                share_kw = dict(sharex=axes[0, 0])
            ax = fig.add_subplot(gs[r, c], **share_kw)
            plot_phase_panel(ax, df, est_names, cmap, title)
            ax.set_xlabel("")
            ax.set_ylabel("")
            axes[r, c] = ax

    # Hide tick labels on interior panels.
    for r in range(n_rows):
        for c in range(3):
            if r < n_rows - 1:
                axes[r, c].tick_params(labelbottom=False)
            if c > 0:
                axes[r, c].tick_params(labelleft=False)

    # Pick a y-step that gives a few ticks per row.
    for r in range(n_rows):
        ymin, ymax = axes[r, 0].get_ylim()
        hi = max(abs(ymin), abs(ymax))
        step = 0.1 if hi > 0.15 else 0.05
        ticks = np.arange(0, hi + step, step)
        ticks = ticks[ticks <= hi + 1e-9]
        axes[r, 0].set_yticks(ticks)
        axes[r, 0].set_yticklabels([f"{t:.2f}" for t in ticks])

    for r, label in enumerate(row_labels):
        axes[r, 0].set_ylabel(f"{label}\n")
    fig.canvas.draw()
    ax_left = axes[0, 0].get_position().x0
    fig.text(ax_left * 0.45, 0.55, r"Correctness predictor $|\mathrm{bias}|$ ($\rightarrow$ better)",
             va="center", ha="center", rotation=90, fontsize=8)
    fig.supxlabel(r"Memorization predictor AUROC ($\rightarrow$ better)")

    # Colorbar.
    cax = fig.add_subplot(gs[:, 3])
    sm = plt.cm.ScalarMappable(cmap=cmap,
                               norm=plt.Normalize(-0.5, len(est_names) - 0.5))
    cbar = fig.colorbar(sm, cax=cax, ticks=range(len(est_names)))
    cbar.ax.set_yticklabels(est_names)

    fig.subplots_adjust(right=0.86, bottom=0.13)
    stem = save_path.with_suffix("")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    plt.close(fig)
    print(f"  Saved {stem}.{{pdf,png}}")


# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="Test set size")
    parser.add_argument("--gamma", type=float, default=0.3,
                        help="Contamination rate")
    parser.add_argument("--n-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha-steps", type=int, default=11,
                        help="Number of grid points for each probe axis")
    parser.add_argument("--difficulty-bins", nargs="+", type=str,
                        default=["easy", "medium", "hard"],
                        help="Difficulty bins for correlated regime")
    parser.add_argument("--model", type=str, default="8b-500b",
                        help="Model tag (e.g. 8b-500b, 1b-500b)")
    parser.add_argument("--benchmark", type=str, default="winogrande_mcq",
                        choices=list(BENCHMARK_BASE_RATES.keys()),
                        help="Benchmark to run simulation on")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip sweeps and re-plot from existing parquet files")
    args = parser.parse_args()

    results_dir = RESULTS_DIR / args.benchmark / args.model
    figures_dir = FIGURES_DIR / args.benchmark / args.model
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    base_rate = BENCHMARK_BASE_RATES[args.benchmark]

    if not args.plot_only:
        print(f"Loading item pool: {args.benchmark} / {args.model}...")
        tag = f'hubble-{args.model}_toks'
        benchmark_dir = EXP50_DIR / args.benchmark
        std_path = benchmark_dir / f'eval_{tag}-standard-hf.parquet'
        prt_path = benchmark_dir / f'eval_{tag}-perturbed-hf.parquet'

        std_df = pd.read_parquet(std_path)
        acc_prefix = 'acc' if f'acc_{tag}-standard-hf' in std_df.columns else 'exact_match'
        confidence_col = f'confidence_{tag}-standard-hf'
        has_confidence = confidence_col in std_df.columns

        full_pool = ItemPool.from_eval_parquets(
            str(std_path), str(prt_path),
            acc_clean_col=f'{acc_prefix}_{tag}-standard-hf',
            acc_perturbed_col=f'{acc_prefix}_{tag}-perturbed-hf',
            confidence_col=confidence_col if has_confidence else None,
        )
        _, pool, _, _ = stratified_split(full_pool)
        print(f"  Items: {pool.n_items} total, "
              f"{len(pool.clean_idx)} clean, {len(pool.contaminated_idx)} contaminated")
        print(f"  Clean accuracy: observed={pool.y_observed.mean():.3f}, "
              f"ground_truth={pool.y_clean.mean():.3f}")

        alphas_m = np.linspace(0.5, 1.0, args.alpha_steps)

        # Derive one bias grid per figure row.
        # p_test = expected clean accuracy on test sets, depends on sampling config.
        # Use the row minimum so panels share an axis.
        p_tests = {}
        for dg in ["low", "mid", "high"]:
            p_tests[f"random_{dg}"] = estimate_p_test(
                pool, args.gamma, dg, "random")
        if has_confidence:
            for db in args.difficulty_bins:
                p_tests[f"correlated_{db}"] = estimate_p_test(
                    pool, args.gamma, "high", "correlated", db)
        for key, pt in p_tests.items():
            print(f"  p_test({key}): {pt:.3f}")

        max_bias_random = min(
            abs(base_rate - p_tests[f"random_{dg}"]) for dg in ["low", "mid", "high"])
        biases_random = np.linspace(0, max_bias_random, args.alpha_steps)
        if has_confidence:
            max_bias_correlated = min(
                abs(base_rate - p_tests[f"correlated_{db}"]) for db in args.difficulty_bins)
            biases_correlated = np.linspace(
                0, max_bias_correlated, args.alpha_steps)
            print(f"  Max bias (random): {max_bias_random:.4f}, "
                  f"(correlated): {max_bias_correlated:.4f}")
        else:
            print(f"  Max bias (random): {max_bias_random:.4f}")
            print(f"  No confidence column — skipping correlated regimes")

    # Row 1: random regime by dose.
    random_results = {}
    for dg in ["low", "mid", "high"]:
        path = results_dir / f"probe_sweep_random_{dg}.parquet"
        if args.plot_only:
            random_results[dg] = pd.read_parquet(path)
        else:
            pt = p_tests[f"random_{dg}"]
            print(f"\n=== Random, dose={dg}, p_test={pt:.3f} ===")
            random_results[dg] = sweep_probe_accuracy(
                pool, alphas_m, biases_random, pt,
                args.gamma, args.n, args.n_replicates, args.seed,
                base_rate=base_rate, dose_group=dg, regime="random",
            )
            random_results[dg].to_parquet(path)

    # Row 2: correlated regime by difficulty bin.
    correlated_results = {}
    has_correlated = (
        results_dir / f"probe_sweep_correlated_{args.difficulty_bins[0]}.parquet").exists()
    if args.plot_only and not has_correlated:
        print("No correlated results found — plotting random only")
    else:
        for db in args.difficulty_bins:
            path = results_dir / f"probe_sweep_correlated_{db}.parquet"
            if args.plot_only:
                correlated_results[db] = pd.read_parquet(path)
            elif has_confidence:
                pt = p_tests[f"correlated_{db}"]
                print(f"\n=== Correlated, bin={db}, p_test={pt:.3f} ===")
                correlated_results[db] = sweep_probe_accuracy(
                    pool, alphas_m, biases_correlated, pt,
                    args.gamma, args.n, args.n_replicates, args.seed,
                    base_rate=base_rate, dose_group="high",
                    regime="correlated", difficulty_bin=db,
                )
                correlated_results[db].to_parquet(path)

    # Combined phase diagram.
    plot_combined_phase(
        random_results, correlated_results,
        figures_dir / "phase_diagram_combined.png",
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
