"""Core primitives for extracting per-token log-probabilities from HF causal LMs.

These generalize hubble.eval.compute_suffix_logprob to return raw per-token
values (instead of averaging), enabling downstream MIA attacks.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedModel, PreTrainedTokenizerBase


@torch.no_grad()
def get_token_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[Tensor, Tensor]:
    """Per-token log-probabilities under a causal LM.

    Returns:
        token_logprobs: shape (seq_len - 1,). ``token_logprobs[i]`` is
            log P(token_ids[i+1] | token_ids[:i+1]).
        token_ids: shape (seq_len - 1,). The predicted token at each position
            (i.e. ``input_ids[1:]``).
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    logits = model(input_ids).logits  # (1, seq_len, vocab)
    log_probs = torch.log_softmax(logits[0, :-1], dim=-1)  # (seq_len-1, vocab)

    target_ids = input_ids[0, 1:]  # (seq_len-1,)
    token_logprobs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_logprobs, target_ids


@torch.no_grad()
def get_full_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-token log-probs plus the full categorical distribution.

    Needed for Min-K%++ which normalizes each token's log-prob by the
    mean and std of the full distribution at that position.

    Returns:
        token_logprobs: shape (seq_len - 1,).
        full_log_probs: shape (seq_len - 1, vocab_size). Full log-softmax.
        token_ids: shape (seq_len - 1,).
    """
    input_ids = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
    input_ids = input_ids.to(model.device)

    logits = model(input_ids).logits
    full_log_probs = torch.log_softmax(logits[0, :-1], dim=-1)  # (seq_len-1, vocab)

    target_ids = input_ids[0, 1:]
    token_logprobs = full_log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    return token_logprobs, full_log_probs, target_ids


@torch.no_grad()
def get_sequence_logprob(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text: str,
) -> float:
    """Mean per-token log-probability (total log-likelihood / n_tokens)."""
    token_logprobs, _ = get_token_logprobs(model, tokenizer, text)
    return token_logprobs.mean().item()
