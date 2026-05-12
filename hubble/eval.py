"""Inference utilities for evaluating Hubble models on benchmark tasks."""

import re
import string

import numpy as np
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MMLU_LETTERS = ["A", "B", "C", "D"]


def load_model(model_id: str, device: str = "cuda", dtype: torch.dtype | None = None):
    """Load a Hubble HF model and tokenizer.

    Args:
        dtype: Override precision. If None, auto-selects: fp32 for 1B models, bf16 for 8B.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if dtype is None:
        dtype = torch.bfloat16 if "8b" in model_id.lower() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=dtype, device_map=device
    )
    model.eval()
    return model, tokenizer


def compute_suffix_logprob(model, tokenizer, prefix: str, suffix: str) -> float:
    """Compute average per-token log-prob of suffix conditioned on prefix.

    Returns:
        Average log-prob per token (negative; higher = more likely).
    """
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
    full_ids = tokenizer.encode(prefix + suffix, add_special_tokens=False)
    # Suffix tokens = everything after the prefix
    suffix_len = len(full_ids) - len(prefix_ids)
    if suffix_len <= 0:
        return float("-inf")

    input_ids = torch.tensor([full_ids], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)

    # Log-probs of each token, shifted: logits[t] predicts token[t+1]
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)
    target_ids = input_ids[0, 1:]

    # Gather log-probs for actual tokens
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    # Average over suffix tokens only
    suffix_log_probs = token_log_probs[-suffix_len:]
    return suffix_log_probs.mean().item()


# ---------------------------------------------------------------------------
# Generic multiple-choice evaluator
# ---------------------------------------------------------------------------

def evaluate_mc_df(model, tokenizer, df: pd.DataFrame, label: str,
                   score_fn, option_names: list[str], answer_offset: int = 0) -> pd.DataFrame:
    """Evaluate a multiple-choice benchmark on a DataFrame.

    Args:
        score_fn: (model, tokenizer, row) -> dict mapping option_name -> logprob.
        option_names: Ordered option names, e.g. ["option1", "option2"] or ["A", "B", "C", "D"].
        answer_offset: Subtracted from df["answer"] to get 0-indexed into option_names.
            WinoGrande uses 1-indexed answers (offset=1); MMLU/PIQA/HellaSwag are 0-indexed (offset=0).

    Adds columns: logprob_{name}_{label}, acc_{label}, confidence_{label}.
    """
    n_options = len(option_names)
    lps = {name: [] for name in option_names}

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Eval {label}"):
        result = score_fn(model, tokenizer, row)
        for name in option_names:
            lps[name].append(result[f"logprob_{name}"])

    df = df.copy()
    lp_arr = np.column_stack([lps[name]
                             for name in option_names])  # (n, n_options)

    for i, name in enumerate(option_names):
        df[f"logprob_{name}_{label}"] = lp_arr[:, i]

    # Prediction: argmax over logprobs
    pred = np.argmax(lp_arr, axis=1)
    answer_idx = df["answer"].values - answer_offset
    df[f"acc_{label}"] = (pred == answer_idx).astype(int)

    # Confidence: LSE-stable softmax, extract correct answer's probability
    lp_max = lp_arr.max(axis=1, keepdims=True)
    exp_shifted = np.exp(lp_arr - lp_max)
    probs = exp_shifted / exp_shifted.sum(axis=1, keepdims=True)
    correct_confidence = probs[np.arange(len(df)), answer_idx]
    df[f"confidence_{label}"] = correct_confidence

    return df


# ---------------------------------------------------------------------------
# Per-benchmark scoring functions
# ---------------------------------------------------------------------------

def _score_winogrande(model, tokenizer, row) -> dict:
    """WinoGrande: split sentence at '_', score suffix logprob for each option."""
    sentence = row["sentence"]
    blank_idx = sentence.index("_")
    before_blank = sentence[:blank_idx]
    after_blank = sentence[blank_idx + 1:]

    return {
        "logprob_option1": compute_suffix_logprob(model, tokenizer, before_blank + row["option1"], after_blank),
        "logprob_option2": compute_suffix_logprob(model, tokenizer, before_blank + row["option2"], after_blank),
    }


def format_mmlu_prompt(question: str, choices: list[str], subject: str) -> str:
    """Format an MMLU question as the standard lm-eval-harness prompt."""
    subject_str = subject.replace("_", " ")
    options = "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(MMLU_LETTERS, choices)
    )
    return (
        f"The following are multiple choice questions (with answers) about {subject_str}.\n\n"
        f"{question.strip()}\n{options}\nAnswer:"
    )


def _score_mmlu(model, tokenizer, row) -> dict:
    """MMLU: score ' A', ' B', ' C', ' D' as continuations of formatted prompt."""
    prompt = format_mmlu_prompt(
        row["question"], row["choices"], row["subject"])
    return {
        f"logprob_{letter}": compute_suffix_logprob(model, tokenizer, prompt, f" {letter}")
        for letter in MMLU_LETTERS
    }


def _score_piqa(model, tokenizer, row) -> dict:
    """PIQA: score sol1/sol2 as continuations of 'Question: {goal}\\nAnswer:'."""
    prefix = f"Question: {row['goal']}\nAnswer:"
    return {
        "logprob_sol1": compute_suffix_logprob(model, tokenizer, prefix, f" {row['sol1']}"),
        "logprob_sol2": compute_suffix_logprob(model, tokenizer, prefix, f" {row['sol2']}"),
    }


def _preprocess_hellaswag(text: str) -> str:
    """Strip WikiHow [title] artifacts, matching lm-eval-harness preprocessing."""
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _score_hellaswag(model, tokenizer, row) -> dict:
    """HellaSwag: score each of 4 endings as continuations of ctx.

    Context is constructed as '{activity_label}: {ctx_a} {ctx_b.capitalize()}'
    to match Hubble's lm-eval-harness config.
    """
    ctx = f"{row['activity_label']}: {row['ctx_a']} {row['ctx_b'].capitalize()}"
    ctx = _preprocess_hellaswag(ctx)
    endings = [_preprocess_hellaswag(e) for e in row["endings"]]
    return {
        f"logprob_ending{i}": compute_suffix_logprob(model, tokenizer, ctx, f" {ending}")
        for i, ending in enumerate(endings)
    }


# ---------------------------------------------------------------------------
# Generative evaluation (e.g. PopQA)
# ---------------------------------------------------------------------------

def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 50) -> str:
    """Greedy-decode an answer, stopping at '\\n\\n' or EOS."""
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    # Decode only newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Truncate at first \n\n (matches lm-eval stop sequence)
    if "\n\n" in text:
        text = text[: text.index("\n\n")]
    return text.strip()


def normalize_answer(s: str) -> str:
    """SQuAD-style normalization: lower, strip, remove articles & punctuation, collapse whitespace."""
    s = s.lower().strip()
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def exact_match_score(prediction: str, answers: list[str]) -> int:
    """1 if normalized prediction matches any acceptable answer, else 0."""
    pred_norm = normalize_answer(prediction)
    return int(any(normalize_answer(a) == pred_norm for a in answers))


def evaluate_gen_df(model, tokenizer, df: pd.DataFrame, label: str,
                    prompt_fn, answer_col: str = "possible_answers",
                    compute_confidence: bool = False) -> pd.DataFrame:
    """Evaluate a generative QA benchmark on a DataFrame.

    Args:
        prompt_fn: (row) -> str prompt to feed the model.
        answer_col: Column containing list of acceptable answer strings.
        compute_confidence: If True, compute mean log-prob of each possible
            answer and store the max as confidence_{label}.

    Adds columns: generated_{label}, exact_match_{label}.
    If compute_confidence: also adds confidence_{label}.
    """
    generated = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Gen {label}"):
        generated.append(generate_answer(model, tokenizer, prompt_fn(row)))

    df = df.copy()
    df[f"generated_{label}"] = generated
    df[f"exact_match_{label}"] = [
        exact_match_score(gen, answers)
        for gen, answers in zip(generated, df[answer_col])
    ]

    if compute_confidence:
        confidences = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Confidence {label}"):
            prompt = prompt_fn(row)
            answers = row[answer_col]
            max_lp = max(
                compute_suffix_logprob(model, tokenizer, prompt, f" {ans}")
                for ans in answers
            )
            confidences.append(max_lp)
        df[f"confidence_{label}"] = confidences

    return df


def _prompt_popqa(row) -> str:
    return f"Question: {row['question']}\nAnswer:"


# ---------------------------------------------------------------------------
# Loss-based evaluation (e.g. Wikipedia passages)
# ---------------------------------------------------------------------------

def compute_text_loss(model, tokenizer, text: str,
                      prefix_tokens: int = 50, suffix_tokens: int = 100) -> float:
    """Compute average per-token loss on a suffix given a prefix.

    Tokenizes the full text, uses the first `prefix_tokens` tokens as context,
    and computes cross-entropy loss on the next `suffix_tokens` tokens.

    Returns:
        Average loss per token (positive; lower = more likely).
        NaN if the text is too short.
    """
    all_ids = tokenizer.encode(text, add_special_tokens=False)
    total_needed = prefix_tokens + suffix_tokens
    if len(all_ids) < total_needed:
        return float("nan")
    # Keep only the tokens we need
    input_ids = torch.tensor([all_ids[:total_needed]], device=model.device)
    with torch.no_grad():
        logits = model(input_ids).logits  # (1, seq_len, vocab)
    # Loss on suffix only: logits[prefix_tokens-1 : total_needed-1] predict
    # tokens[prefix_tokens : total_needed]
    suffix_logits = logits[0, prefix_tokens - 1 : total_needed - 1]
    suffix_labels = input_ids[0, prefix_tokens : total_needed]
    loss = torch.nn.functional.cross_entropy(
        suffix_logits, suffix_labels, reduction="mean"
    )
    return loss.item()


def evaluate_loss_df(model, tokenizer, df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Evaluate a text passage dataset by computing per-item loss.

    Uses the first 50 tokens as prefix context, then computes loss on the
    next 100 tokens.

    Adds column: loss_{label}.
    """
    losses = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Loss {label}"):
        losses.append(compute_text_loss(model, tokenizer, row["text"]))

    df = df.copy()
    df[f"loss_{label}"] = losses
    return df


# ---------------------------------------------------------------------------
# Benchmark registry & unified entry point
# ---------------------------------------------------------------------------

def evaluate_benchmark_df(model, tokenizer, df: pd.DataFrame, label: str, benchmark: str) -> pd.DataFrame:
    """Evaluate a benchmark on a DataFrame.

    Args:
        benchmark: "winogrande", "mmlu", "piqa", "hellaswag", or "popqa".

    For MC benchmarks, adds: logprob_{option}_{label}, acc_{label}, confidence_{label}.
    For generative benchmarks, adds: generated_{label}, exact_match_{label}.
    """
    if benchmark == "winogrande":
        return evaluate_mc_df(model, tokenizer, df, label, _score_winogrande, ["option1", "option2"], answer_offset=1)
    elif benchmark == "mmlu":
        return evaluate_mc_df(model, tokenizer, df, label, _score_mmlu, MMLU_LETTERS, answer_offset=0)
    elif benchmark == "piqa":
        return evaluate_mc_df(model, tokenizer, df, label, _score_piqa, ["sol1", "sol2"], answer_offset=0)
    elif benchmark == "hellaswag":
        return evaluate_mc_df(model, tokenizer, df, label, _score_hellaswag, ["ending0", "ending1", "ending2", "ending3"], answer_offset=0)
    elif benchmark == "popqa":
        return evaluate_gen_df(model, tokenizer, df, label, _prompt_popqa,
                               answer_col="possible_answers", compute_confidence=True)
    elif benchmark == "wikipedia":
        return evaluate_loss_df(model, tokenizer, df, label)
    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")
