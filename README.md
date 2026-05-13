# Spiking the training data to correct for test set contamination

The literature on test set contamination has largely focused on detection, but statistical correction of contaminated test scores is underexplored. Our core proposal is to *spike* the training data and intentionally contaminate some test examples at known rates. The spiked examples can then be used to calibrate memorization detectors which enable principled statistical correction. To evaluate different correction estimators, we first present a simulation framework based on the Hubble models. Hubble models come in minimal pairs, where the perturbed model was intentionally contaminated with several test sets, while the standard model was not, serving as the counterfactual and correction target. We consider estimators that use information from a memorization predictor, a correctness predictor, or both. In simulation, we establish basic statistical intuitions and show that estimators leveraging memorization and correctness information are better than naive estimation which makes no correction at all. We then instantiate several memorization and correctness predictors, and find that simple predictors such as Platt-scaled membership inference metrics provide good signal for correction. Finally, we examine the practical considerations of spiking. Simple memorization predictors need no more than 10 examples for calibration and decently transfer from one dataset to another. Taken together, spiking is a promising solution to test set contamination.


## Overview

This repository contains the experiment code for generating Hubble benchmark caches, fitting memorization and correctness predictors, running adjustment simulations, and producing sample-efficiency / predictor-transfer outputs.


## Setup

Create a `uv` environment and install the dependencies:

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
export PYTHONPATH="$PWD"
```

This release contains the code used for the experiments in our paper. The
data-generation and training commands assume access to the required Hugging
Face model weights and sufficient GPU resources. If you need gated models such
as Llama or Qwen, log in first:

```bash
huggingface-cli login
```

The scripts write results next to their module, usually under
`spiking/<module>/results/` and figures under `spiking/<module>/figures/`.

## Run Everything

To run all paper experiment stages in order, use the main pipeline under
`spiking`:

```bash
uv run python spiking/main.py
```

Useful options:

```bash
uv run python spiking/main.py --dry-run
uv run python spiking/main.py --scope main --skip-gpu
uv run python spiking/main.py --stage memorization correctness adjustment
```

## Data Generation

These scripts create the shared caches used by the downstream predictors.

Evaluate the Hubble models on all benchmarks:

```bash
uv run python spiking/data_generation/run_evals.py
```

For the main paper setting, the downstream modules use the 8B/500B standard and
perturbed Hubble pair. Running the command above creates any missing evaluation
caches for the available Hubble models.

Compute memorization attack scores for the 8B/500B perturbed model. In this
script, `--model 3` selects the 8B/500B perturbed model from the perturbed-model
list.

```bash
for attack in loss zlib min_k min_k_plus_plus reference; do
  uv run python spiking/data_generation/run_mia_scores.py score \
    --attack "$attack" \
    --model 3
done

uv run python spiking/data_generation/run_mia_scores.py combine \
  --model-filter 8b-500b
```

Extract hidden-state features for memorization predictors:

```bash
uv run python spiking/data_generation/run_hidden_states.py extract --model 3
uv run python spiking/data_generation/run_hidden_states.py verify
```

Extract external LLM confidence caches for correctness predictors:

```bash
uv run python spiking/data_generation/run_llm_confidence.py extract --external llama
uv run python spiking/data_generation/run_llm_confidence.py extract --external pythia --size 6.9b
uv run python spiking/data_generation/run_llm_confidence.py extract --external qwen --size 8b

uv run python spiking/data_generation/run_llm_confidence.py verify --external llama
uv run python spiking/data_generation/run_llm_confidence.py verify --external pythia --size 6.9b
uv run python spiking/data_generation/run_llm_confidence.py verify --external qwen --size 8b
```

## Memorization Predictors

Fit memorization predictors from the generated MIA score and feature caches:

```bash
uv run python spiking/memorization/run.py
```

Run a single benchmark while debugging:

```bash
uv run python spiking/memorization/run.py --benchmark mmlu
```

Run the memorization-only simulation using cached `d_hat` predictions:

```bash
uv run python spiking/memorization/run_simulation.py
```

Useful smaller run:

```bash
uv run python spiking/memorization/run_simulation.py \
  --benchmark mmlu \
  --n-replicates 100
```

## Correctness / Correction Predictors

The correctness predictors produce `c_hat` predictions. RoBERTa requires training
and should be run on a GPU.

Train RoBERTa correctness predictors:

```bash
uv run python spiking/correctness/run_roberta.py \
  --question-only \
  --lr 5e-6 \
  --freeze-layers 0 5
```

For a quick single-benchmark training run:

```bash
uv run python spiking/correctness/run_roberta.py \
  --benchmark mmlu \
  --epochs 1 \
  --question-only
```

Generate Platt-scaled external LLM correctness predictors from the confidence
caches:

```bash
uv run python spiking/correctness/run_external_llm.py --external llama
uv run python spiking/correctness/run_external_llm.py --external pythia --size 6.9b
uv run python spiking/correctness/run_external_llm.py --external qwen --size 8b
```

Evaluate cached correctness predictors together:

```bash
uv run python spiking/correctness/run_evals.py \
  --pythia-size 6.9b \
  --qwen-size 8b \
  --question-only
```

Run the correctness simulation:

```bash
uv run python spiking/correctness/run_simulation.py \
  --pythia-size 6.9b \
  --question-only
```

## Adjustment Simulation

The adjustment simulation combines memorization `d_hat` and correctness `c_hat`
outputs. Run memorization and correctness predictors first.

```bash
uv run python spiking/adjustment/run_simulation.py
```

A smaller debug run:

```bash
uv run python spiking/adjustment/run_simulation.py \
  --benchmark mmlu \
  --n-replicates 100
```

Plot calibration bars from a completed run. The default full-run suffix is:

```bash
uv run python spiking/adjustment/plot_calibration.py \
  --suffix all_8b-500b_min_k_plus_plus_standard_n500_g0.3_r1000
```

## Sample Efficiency

The sample-efficiency plotting scripts expect precomputed files like:

```text
spiking/sample_efficiency/results/sample_efficiency_high.parquet
spiking/sample_efficiency/results/sample_efficiency_mid.parquet
```

Plot the default high/mid dose grid:

```bash
uv run python spiking/sample_efficiency/plot.py
```

Plot a custom subset:

```bash
uv run python spiking/sample_efficiency/plot.py \
  --dose-groups high mid low \
  --benchmarks winogrande_mcq mmlu popqa
```

## Predictor Transfer

Predictor transfer experiments reuse the generated `d_hat`, `c_hat`, MIA score, and
hidden-state caches.

Run memorization-predictor transfer:

```bash
uv run python spiking/predictor_transfer/run_mem_sim.py --regime both
```

Run correctness-predictor transfer:

```bash
uv run python spiking/predictor_transfer/run_corr_sim.py --regime both
```

Visualize memorization transfer:

```bash
uv run python spiking/predictor_transfer/visualize.py \
  --mode mem \
  --dose-group all
```

Visualize correctness transfer:

```bash
uv run python spiking/predictor_transfer/visualize.py \
  --mode corr \
  --dose-group all
```

For quicker debugging, restrict transfer runs to one source and target:

```bash
uv run python spiking/predictor_transfer/run_mem_sim.py \
  --source mmlu \
  --target piqa \
  --regime random \
  --n-replicates 100

uv run python spiking/predictor_transfer/run_corr_sim.py \
  --source mmlu \
  --target piqa \
  --regime random \
  --n-replicates 100
```

## Optional Synthetic Simulation

The synthetic phase-diagram simulation is CPU-only once evaluation caches exist:

```bash
uv run python spiking/simulation/run.py --benchmark mmlu
```

Generate appendix figures from saved simulation parquet files:

```bash
uv run python spiking/simulation/create_appendix.py
```
