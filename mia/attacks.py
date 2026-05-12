"""Membership inference attack scoring functions.

Each function takes (model, tokenizer, text, **kwargs) -> float.
Convention: **higher score = more likely a training member (memorized)**.

References:
    - LOSS: Yeom et al. (2018), "Privacy Risk in Machine Learning"
    - Zlib: Carlini et al. (2021), "Extracting Training Data from Large LMs"
    - Min-K%: Shi et al. (2023), "Detecting Pretraining Data from Large LMs"
    - Min-K%++: Zhang et al. (2024), github.com/zjysteven/mink-plus-plus
    - Reference: Carlini et al. (2021)
    - GradNorm: based on gradient-based MI from MIMIR
"""

from __future__ import annotations

import zlib as _zlib

import numpy as np
import torch
import torch.nn.functional as F

from mia.logprobs import get_full_logprobs, get_sequence_logprob, get_token_logprobs


# -- Simple attacks (blackbox, single forward pass) --------------------------


def loss(model, tokenizer, text: str) -> float:
    """LOSS attack: mean per-token log-likelihood.

    Higher (less negative) = model assigns higher probability = more likely memorized.
    """
    return get_sequence_logprob(model, tokenizer, text)


def zlib(model, tokenizer, text: str) -> float:
    """Zlib attack: mean LL normalized by zlib-compressed byte length.

    Calibrates model confidence by intrinsic text complexity.
    """
    ll = get_sequence_logprob(model, tokenizer, text)
    zlib_len = len(_zlib.compress(text.encode("utf-8")))
    return ll / zlib_len


def min_k(model, tokenizer, text: str, *, k: float = 0.2,
          window: int = 1, stride: int = 1) -> float:
    """Min-K% Prob attack: average of bottom-k% token log-probs.

    Intuition: memorized text has fewer "surprising" tokens, so even the
    least-likely k% tokens have relatively high probability.

    Higher (less negative) mean = more memorized.
    """
    token_lps, _ = get_token_logprobs(model, tokenizer, text)

    # N-gram windowing: average log-probs within each window
    if window > 1:
        ngram_probs = []
        for i in range(0, len(token_lps) - window + 1, stride):
            ngram_probs.append(token_lps[i:i + window].mean().item())
        values = np.array(ngram_probs)
    else:
        values = token_lps.float().cpu().numpy()

    values.sort()
    n_bottom = max(1, int(len(values) * k))
    return float(values[:n_bottom].mean())


def min_k_plus_plus(model, tokenizer, text: str, *, k: float = 0.2) -> float:
    """Min-K%++ attack: z-score normalized min-k%.

    At each token position, normalizes log P(token) by the mean and std
    of the full categorical distribution, then takes the bottom k%.
    """
    token_lps, full_lps, _ = get_full_logprobs(model, tokenizer, text)

    # Per-position mean and std of log-probs under the categorical dist
    # mu_i = E_{v~p_i}[log p_i(v)] = sum_v p_i(v) * log p_i(v)
    probs = full_lps.exp()  # (seq_len-1, vocab)
    mu = (probs * full_lps).sum(dim=-1)  # (seq_len-1,)
    sigma_sq = (probs * full_lps.square()).sum(dim=-1) - mu.square()
    sigma = sigma_sq.clamp(min=1e-10).sqrt()

    z_scores = ((token_lps - mu) / sigma).float().cpu().numpy()
    z_scores.sort()
    n_bottom = max(1, int(len(z_scores) * k))
    return float(z_scores[:n_bottom].mean())


# -- Attacks requiring additional resources -----------------------------------


def reference(model, tokenizer, text: str, *,
              ref_model, ref_tokenizer=None) -> float:
    """Reference attack: LL difference between target and reference model.

    Positive score means target model assigns higher likelihood than the
    reference, suggesting the text was in the target's training data.

    Args:
        ref_model: Reference model (e.g., a different checkpoint or smaller model).
        ref_tokenizer: Tokenizer for ref_model. Defaults to ``tokenizer`` if None.
    """
    if ref_tokenizer is None:
        ref_tokenizer = tokenizer
    ll_target = get_sequence_logprob(model, tokenizer, text)
    ll_ref = get_sequence_logprob(ref_model, ref_tokenizer, text)
    return ll_target - ll_ref


def gradnorm(model, tokenizer, text: str, *, p: float = float("inf")) -> float:
    """GradNorm attack: p-norm of loss gradients w.r.t. model parameters.

    Lower gradient norm suggests the model has "converged" on this text
    (i.e., memorized it). Returns the negated norm so higher = more memorized.

    NOTE: This is the only attack that requires gradients. The model must
    NOT be wrapped in torch.no_grad(). This function handles eval() and
    zero_grad() internally.
    """
    input_ids = tokenizer.encode(
        text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    model.eval()
    model.zero_grad()
    logits = model(input_ids).logits
    shift_logits = logits[0, :-1]
    shift_labels = input_ids[0, 1:]
    ntp_loss = F.cross_entropy(shift_logits, shift_labels)
    ntp_loss.backward()

    grad_norms = []
    for param in model.parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.detach().norm(p))
    score = -torch.stack(grad_norms).mean().item()

    model.zero_grad()
    return score


# -- Registry for convenience ------------------------------------------------

ATTACKS = {
    "loss": loss,
    "zlib": zlib,
    "min_k": min_k,
    "min_k_plus_plus": min_k_plus_plus,
    "reference": reference,
    "gradnorm": gradnorm,
}

#: Attacks that only need a single model (no ref model, no gradients).
SIMPLE_ATTACKS = ["loss", "zlib", "min_k", "min_k_plus_plus"]
