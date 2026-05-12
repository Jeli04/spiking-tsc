"""Memorization probes: detect P(contaminated | x).

Three probe families:
  - MIAProbe: Platt-scaled logistic regression on pre-computed MIA attack
    scores (CPU only, no model needed).
  - HiddenStateProbe: Extract pooled hidden states from a transformer layer,
    fit logistic regression (GPU for extraction, CPU for fitting).
  - ResidualProbe: Extract pooled residual stream from a specific transformer
    layer via hooks, fit logistic regression (GPU for extraction, CPU for fitting).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# ==========================================================================
# Base class
# ==========================================================================

class Probe(ABC):
    """Base class for all probes (memorization and correctness).

    All probes expose a unified fit/predict interface. Subclasses accept
    pre-computed features (numpy arrays) or DataFrames, depending on type.
    """

    @property
    @abstractmethod
    def feature_key(self) -> str:
        """Unique key for caching and result naming."""
        ...

    @abstractmethod
    def fit(self, data, labels, **kwargs) -> None:
        """Train the probe on pre-computed features and labels."""
        ...

    @abstractmethod
    def predict(self, data, **kwargs) -> np.ndarray:
        """Return predicted class labels."""
        ...

    @abstractmethod
    def predict_proba(self, data, **kwargs) -> np.ndarray:
        """Return P(contaminated) for each item (1D array)."""
        ...


# ==========================================================================
# MIA probes (pre-computed scores + Platt scaling)
# ==========================================================================

class MIAProbe(Probe):
    """Platt-scaled memorization probe on pre-computed MIA attack scores.

    NOTE: [pedagogical] Platt scaling converts a raw score into a calibrated
    probability by fitting a logistic regression on the score as a single
    feature. This gives us P(contaminated | score) rather than a hard threshold.

    Accepts either a DataFrame with a score column (df[score_col]) or raw
    numpy arrays passed directly to fit/predict/predict_proba.

    Usage::

        probe = MIAProbe('min_k_plus_plus')
        probe.fit(scores_cal, labels_cal)
        d_hat = probe.predict_proba(scores_sim)
    """

    def __init__(self, score_col: str):
        self.score_col = score_col
        self._lr: LogisticRegression | None = None

    @property
    def feature_key(self) -> str:
        return f'mia_{self.score_col}'

    def _get_scores(self, data) -> np.ndarray:
        """Extract scores from DataFrame column or pass through numpy array."""
        if isinstance(data, pd.DataFrame):
            return data[self.score_col].values
        return np.asarray(data)

    def fit(self, data, labels, **kwargs) -> None:
        """Fit Platt scaling on calibration scores and binary labels."""
        scores = self._get_scores(data)
        self._lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._lr.fit(scores.reshape(-1, 1), labels)

    def predict(self, data, **kwargs) -> np.ndarray:
        """Return binary predictions at 0.5 threshold."""
        return (self.predict_proba(data) >= 0.5).astype(int)

    def predict_proba(self, data, **kwargs) -> np.ndarray:
        """Return calibrated P(contaminated) for each item."""
        assert self._lr is not None, 'Call fit() first'
        scores = self._get_scores(data)
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]


# ==========================================================================
# Hidden state probes (frozen features + sklearn)
# ==========================================================================

class HiddenStateProbe(Probe):
    """Probe on pooled hidden states from a specific transformer layer.

    Two-phase usage:
      1. extract_features() — GPU. Extract and cache pooled hidden states.
      2. fit() / predict_proba() — CPU. Operate on pre-extracted numpy arrays.

    Subclasses set `layer` and `pool`.

    Usage::

        probe = FinalLayerLinear()

        # Phase 1: extract (GPU, typically cached by exp 12)
        probe.extract_features(model, tokenizer, texts, cache_path=path)

        # Phase 2: fit + predict (CPU, on pre-loaded arrays)
        features = np.load(path)['hidden_states']
        probe.fit(features[cal_idx], labels_cal)
        d_hat = probe.predict_proba(features[sim_idx])
    """

    layer: int = -1

    def __init__(self, seed: int = 42):
        self.seed = seed
        self._clf = None

    @staticmethod
    def mean(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # NOTE: [shape] mask: batch x seq_len -> unsqueeze to batch x seq_len x 1
        # for broadcasting against hidden: batch x seq_len x hidden_dim
        mask_expanded = mask.unsqueeze(-1).float()
        return (hidden * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)

    @staticmethod
    def last(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        seq_lengths = mask.sum(dim=1) - 1
        return hidden[torch.arange(hidden.size(0)), seq_lengths]

    pool = mean

    @property
    def feature_key(self) -> str:
        return f'layer{self.layer}_pool{self.pool.__name__}'

    # ------------------------------------------------------------------
    # Phase 1: Feature extraction (GPU)
    # ------------------------------------------------------------------

    def extract_features(
        self,
        model,
        tokenizer,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        cache_path: Path | None = None,
    ) -> np.ndarray:
        """Extract pooled hidden state features for all texts.

        If cache_path is provided, loads from cache if it exists,
        otherwise extracts and saves to cache.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f'[{self.feature_key}] Loading cached hidden states')
                return np.load(cache_path)['hidden_states']

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f'[{self.feature_key}] Extracting features...')
        results = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc='Extracting hidden states')

        for start in iterator:
            batch_texts = texts[start: start + batch_size]
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt',
            ).to(model.device)

            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)

            hidden = outputs.hidden_states[self.layer]
            mask = inputs['attention_mask']
            pooled = self.pool(hidden, mask)
            results.append(pooled.cpu().float().numpy())

        X = np.concatenate(results, axis=0)
        if cache_path is not None:
            np.savez(cache_path, hidden_states=X)
            print(f'[{self.feature_key}] Cached (shape={X.shape})')
        return X

    # ------------------------------------------------------------------
    # Phase 2: Fit / predict (CPU, on pre-extracted features)
    # ------------------------------------------------------------------

    def fit(self, data, labels, **kwargs) -> None:
        """Fit logistic regression on pre-extracted feature array."""
        X = np.asarray(data)
        self._clf = make_pipeline(
            StandardScaler(),
            LogisticRegressionCV(
                Cs=10, cv=5, solver='lbfgs', max_iter=1000,
                scoring='accuracy', class_weight='balanced',
                random_state=self.seed, n_jobs=-1,
            ),
        )
        self._clf.fit(X, labels)

    def predict(self, data, **kwargs) -> np.ndarray:
        assert self._clf is not None, 'Call fit() first'
        return self._clf.predict(np.asarray(data))

    def predict_proba(self, data, **kwargs) -> np.ndarray:
        """Return P(contaminated) for each item (1D array)."""
        assert self._clf is not None, 'Call fit() first'
        return self._clf.predict_proba(np.asarray(data))[:, 1]


class FinalLayerLinear(HiddenStateProbe):
    """Last hidden layer, mean pool, logistic regression."""

    layer = -1
    pool = staticmethod(HiddenStateProbe.mean)


class FinalLayerLastTokenLinear(HiddenStateProbe):
    """Last hidden layer, last-token pool, logistic regression."""

    layer = -1
    pool = staticmethod(HiddenStateProbe.last)


# ==========================================================================
# Residual stream probes (hook-based extraction + sklearn)
# ==========================================================================

DEFAULT_RESIDUAL_LAYERS = list(range(26, 36))


class ResidualProbe(HiddenStateProbe):
    """Probe on pooled residual stream from a specific transformer layer.

    Unlike HiddenStateProbe which uses ``output_hidden_states``, this probe
    hooks into ``layer.post_attention_layernorm`` to capture the residual
    stream input (same hook point as exp 06).

    Usage::

        probe = ResidualProbe(layer=30, pool_name='mean')

        # Phase 1: extract (GPU)
        probe.extract_features(model, tokenizer, texts, cache_path=path)

        # Phase 2: fit + predict (CPU)
        features = np.load(path)['hidden_states']
        probe.fit(features[cal_idx], labels_cal)
        d_hat = probe.predict_proba(features[sim_idx])
    """

    def __init__(self, layer: int = 35, pool_name: str = 'mean', seed: int = 42):
        super().__init__(seed=seed)
        self._layer = layer
        self._pool_name = pool_name
        if pool_name == 'mean':
            self._pool_fn = HiddenStateProbe.mean
        elif pool_name == 'last':
            self._pool_fn = HiddenStateProbe.last
        else:
            raise ValueError(f'Unknown pool_name: {pool_name!r}')

    @property
    def feature_key(self) -> str:
        return f'residual_L{self._layer}_{self._pool_name}'

    def extract_features(
        self,
        model,
        tokenizer,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = True,
        cache_path: Path | None = None,
    ) -> np.ndarray:
        """Extract pooled residual stream features for all texts.

        Hooks into ``layer.post_attention_layernorm`` to capture the residual
        stream (input to the layernorm, i.e. the residual before MLP).
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                print(f'[{self.feature_key}] Loading cached residuals')
                return np.load(cache_path)['hidden_states']

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print(f'[{self.feature_key}] Extracting residual features...')
        transformer = model.model
        target_layer = transformer.layers[self._layer]

        captured = []

        def hook_fn(module, args, output):
            residual = args[0] if isinstance(args, tuple) else args
            captured.append(residual.detach())

        handle = target_layer.post_attention_layernorm.register_forward_hook(hook_fn)

        results = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc=f'Extracting residual L{self._layer}')

        try:
            for start in iterator:
                batch_texts = texts[start: start + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                ).to(model.device)

                with torch.no_grad():
                    model(**inputs)

                residual = captured.pop()
                mask = inputs['attention_mask']
                pooled = self._pool_fn(residual, mask)
                results.append(pooled.cpu().float().numpy())
        finally:
            handle.remove()

        X = np.concatenate(results, axis=0)
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, hidden_states=X)
            print(f'[{self.feature_key}] Cached (shape={X.shape})')
        return X

    @staticmethod
    def extract_all_layers(
        model,
        tokenizer,
        texts: list[str],
        layers: list[int] | None = None,
        pool_names: list[str] | None = None,
        batch_size: int = 32,
        cache_dir: Path | None = None,
        show_progress: bool = True,
    ) -> dict[tuple[int, str], np.ndarray]:
        """Extract residual features for multiple layers and poolings in one pass.

        More efficient than calling extract_features() per layer since the model
        forward pass is shared across all hooked layers.

        Returns dict mapping (layer, pool_name) -> features array.
        """
        if layers is None:
            layers = DEFAULT_RESIDUAL_LAYERS
        if pool_names is None:
            pool_names = ['mean', 'last']

        # Check which (layer, pool) combos need extraction
        needed = {}
        cached = {}
        for layer in layers:
            for pool_name in pool_names:
                key = (layer, pool_name)
                if cache_dir is not None:
                    cache_path = cache_dir / f'features_layer{layer:02d}_{pool_name}.npz'
                    if cache_path.exists():
                        cached[key] = np.load(cache_path)['hidden_states']
                        continue
                needed[key] = True

        if not needed:
            print(f'All {len(cached)} residual caches found, skipping extraction')
            return cached

        needed_layers = sorted({l for l, _ in needed})
        print(f'Extracting residual layers {needed_layers} ({len(needed)} combos)')

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        transformer = model.model

        # Storage: layer_idx -> list of (batch, seq_len, hidden_dim) tensors
        storage: dict[int, list] = defaultdict(list)
        seq_lengths: list[list[int]] = []

        def make_hook(layer_idx):
            def hook_fn(module, args, output):
                residual = args[0] if isinstance(args, tuple) else args
                storage[layer_idx].append(residual.detach().cpu())
            return hook_fn

        handles = []
        for layer_idx in needed_layers:
            h = transformer.layers[layer_idx].post_attention_layernorm.register_forward_hook(
                make_hook(layer_idx))
            handles.append(h)

        n_batches = (len(texts) + batch_size - 1) // batch_size
        iterator = range(0, len(texts), batch_size)
        if show_progress:
            iterator = tqdm(iterator, total=n_batches,
                            desc='Extracting residuals (all layers)')

        try:
            for start in iterator:
                batch_texts = texts[start: start + batch_size]
                inputs = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt',
                ).to(model.device)

                with torch.no_grad():
                    model(**inputs)

                mask = inputs['attention_mask']
                batch_seq_lens = mask.sum(dim=1).tolist()
                seq_lengths.append(batch_seq_lens)
        finally:
            for h in handles:
                h.remove()

        # Pool and save
        pool_fns = {
            'mean': HiddenStateProbe.mean,
            'last': HiddenStateProbe.last,
        }

        results = dict(cached)
        for layer_idx in needed_layers:
            batch_tensors = storage[layer_idx]
            # Reconstruct attention masks for pooling
            for pool_name in pool_names:
                if (layer_idx, pool_name) not in needed:
                    continue
                pooled_batches = []
                for bi, batch_tensor in enumerate(batch_tensors):
                    bs = batch_tensor.shape[0]
                    seq_len = batch_tensor.shape[1]
                    # Rebuild mask from seq_lengths
                    batch_lens = seq_lengths[bi]
                    mask = torch.zeros(bs, seq_len, dtype=torch.long)
                    for j, length in enumerate(batch_lens):
                        mask[j, :length] = 1
                    pooled = pool_fns[pool_name](batch_tensor, mask)
                    pooled_batches.append(pooled.float().numpy())
                X = np.concatenate(pooled_batches, axis=0)
                results[(layer_idx, pool_name)] = X

                if cache_dir is not None:
                    cache_path = cache_dir / f'features_layer{layer_idx:02d}_{pool_name}.npz'
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache_path, hidden_states=X)
                    print(f'  Cached residual L{layer_idx}_{pool_name} (shape={X.shape})')

        return results


# ==========================================================================
# Registry
# ==========================================================================

PROBES: dict[str, type[HiddenStateProbe]] = {
    'final_layer_linear': FinalLayerLinear,
    'final_layer_last_token_linear': FinalLayerLastTokenLinear,
}

RESIDUAL_PROBES: dict[str, tuple[int, str]] = {}
for _layer in DEFAULT_RESIDUAL_LAYERS:
    for _pool in ('mean', 'last'):
        RESIDUAL_PROBES[f'residual_L{_layer}_{_pool}'] = (_layer, _pool)
