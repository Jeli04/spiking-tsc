"""Plot accuracy vs. duplication count for all benchmarks.

Reproduces the Hubble accuracy-by-contamination figures, generalized across
benchmarks. Handles both MC benchmarks (acc_) and generative (exact_match_).

Usage:
  uv run python src/spiking/data_generation/plot_evals.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hubble.runner import HUBBLE_MODELS, _model_label

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

BENCHMARKS = ["winogrande", "mmlu", "piqa", "popqa", "hellaswag"]


def _parse_config(model_id):
    """Parse a Hubble model id into size, variant, and token scale."""
    parts = model_id.split("/")[-1].split("-")
    size, scale, variant = parts[1], parts[2].replace("_toks", ""), parts[3]
    return (size, variant, scale)


MODEL_LABELS = {_parse_config(m): _model_label(m) for m in HUBBLE_MODELS}


def _acc_col(label, benchmark):
    """Return the accuracy column name for a given model label and benchmark."""
    if benchmark == "popqa":
        return f"exact_match_{label}"
    return f"acc_{label}"


def load_benchmark(benchmark: str) -> pd.DataFrame | None:
    """Load and merge all available per-model eval parquets for a benchmark."""
    bench_dir = RESULTS_DIR / benchmark
    if not bench_dir.exists():
        return None

    # Find all eval parquets.
    parquets = sorted(bench_dir.glob("eval_*.parquet"))
    if not parquets:
        return None

    # Start from the first file and add columns from the rest.
    df = pd.read_parquet(parquets[0])
    base_cols = [c for c in df.columns if not any(
        c.startswith(p) for p in ("acc_", "confidence_", "logprob_", "exact_match_", "generated_")
    )]

    for pq in parquets[1:]:
        other = pd.read_parquet(pq)
        new_cols = [c for c in other.columns if c not in base_cols and c not in df.columns]
        for col in new_cols:
            df[col] = other[col]

    return df


# Display config: (variant, scale) -> (color, marker, linestyle)
LINE_STYLES = {
    ("standard", "100b"): ("tab:blue", "o", "--"),
    ("perturbed", "100b"): ("tab:red", "o", "--"),
    ("standard", "500b"): ("tab:blue", "s", "-"),
    ("perturbed", "500b"): ("tab:red", "s", "-"),
}

DISPLAY_LABELS = {
    ("standard", "100b"): "Standard 100B",
    ("perturbed", "100b"): "Perturbed 100B",
    ("standard", "500b"): "Standard 500B",
    ("perturbed", "500b"): "Perturbed 500B",
}

# Chance baselines per benchmark.
CHANCE = {
    "winogrande": 0.5,
    "mmlu": 0.25,
    "piqa": 0.5,
    "hellaswag": 0.25,
    "popqa": None,  # generative benchmark
}


def plot_accuracy_by_duplicates(benchmark: str, df: pd.DataFrame):
    """Plot accuracy vs. duplication count: 1b and 8b side by side."""
    # Match the Hubble figures for WinoGrande.
    if benchmark == "winogrande" and "format" in df.columns:
        df = df[df["format"] == "infill"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, size in zip(axes, ["1b", "8b"]):
        for variant, scale in [("standard", "100b"), ("perturbed", "100b"),
                                ("standard", "500b"), ("perturbed", "500b")]:
            label = MODEL_LABELS[(size, variant, scale)]
            col = _acc_col(label, benchmark)
            if col not in df.columns:
                continue

            color, marker, ls = LINE_STYLES[(variant, scale)]
            display = DISPLAY_LABELS[(variant, scale)]

            grouped = df.groupby("duplicates")[col]
            p = grouped.mean()
            n = grouped.count()
            se = np.sqrt(p * (1 - p) / n)
            ax.errorbar(p.index.astype(str), p, yerr=1.96 * se,
                        fmt=marker, linestyle=ls, label=display,
                        color=color, capsize=3)

        ax.set_xlabel("Duplication count")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{benchmark} ({size.upper()})")
        ax.legend()

        chance = CHANCE.get(benchmark)
        if chance is not None:
            ax.axhline(chance, color="gray", linestyle="--", alpha=0.3)
            ax.set_ylim(max(0, chance - 0.1), 1.05)

    bench_label = "Infill" if benchmark == "winogrande" else benchmark.upper()
    fig.suptitle(f"{bench_label} — Accuracy by Contamination Level", fontsize=13)
    fig.tight_layout()
    out = FIGURES_DIR / f"accuracy_by_duplication_{benchmark}.png"
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")
    plt.close(fig)


def plot_all_benchmarks_combined(benchmark_dfs: dict[str, pd.DataFrame]):
    """Single figure with all available benchmarks as rows, 1b/8b as columns."""
    available = [b for b in BENCHMARKS if b in benchmark_dfs]
    if not available:
        print("No benchmark data available for combined plot.")
        return

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows), squeeze=False)

    for row, benchmark in enumerate(available):
        df = benchmark_dfs[benchmark]
        if benchmark == "winogrande" and "format" in df.columns:
            df = df[df["format"] == "infill"]

        for col_idx, size in enumerate(["1b", "8b"]):
            ax = axes[row, col_idx]
            for variant, scale in [("standard", "100b"), ("perturbed", "100b"),
                                    ("standard", "500b"), ("perturbed", "500b")]:
                label = MODEL_LABELS[(size, variant, scale)]
                acc = _acc_col(label, benchmark)
                if acc not in df.columns:
                    continue

                color, marker, ls = LINE_STYLES[(variant, scale)]
                display = DISPLAY_LABELS[(variant, scale)]

                grouped = df.groupby("duplicates")[acc]
                p = grouped.mean()
                n = grouped.count()
                se = np.sqrt(p * (1 - p) / n)
                ax.errorbar(p.index.astype(str), p, yerr=1.96 * se,
                            fmt=marker, linestyle=ls, label=display,
                            color=color, capsize=3)

            chance = CHANCE.get(benchmark)
            if chance is not None:
                ax.axhline(chance, color="gray", linestyle="--", alpha=0.3)

            ax.set_xlabel("Duplication count")
            if col_idx == 0:
                ax.set_ylabel("Accuracy")
            ax.set_title(f"{benchmark} ({size.upper()})")
            if row == 0 and col_idx == 1:
                ax.legend(fontsize=8)

    fig.suptitle("Accuracy by Contamination Level — All Benchmarks", fontsize=14, y=1.01)
    fig.tight_layout()
    out = FIGURES_DIR / "accuracy_by_duplication_all.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


def main():
    benchmark_dfs = {}
    for benchmark in BENCHMARKS:
        df = load_benchmark(benchmark)
        if df is not None:
            print(f"Loaded {benchmark}: {len(df)} examples, "
                  f"{sum(1 for c in df.columns if c.startswith(('acc_', 'exact_match_')))} acc columns")
            benchmark_dfs[benchmark] = df
            plot_accuracy_by_duplicates(benchmark, df)
        else:
            print(f"No data for {benchmark}, skipping.")

    plot_all_benchmarks_combined(benchmark_dfs)
    print("\nDone.")


if __name__ == "__main__":
    main()
