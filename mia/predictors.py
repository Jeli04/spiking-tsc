"""MIA predictors: threshold-based binary classifiers over attack scores."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm

from hubble.predictors import Predictor


class AttackPredictor(Predictor):
    """Predictor that wraps any attack function from ``mia.attacks``.

    Computes MIA scores on the fly by running model inference on a text
    column of the DataFrame.

    Example::

        from mia.attacks import min_k
        predictor = AttackPredictor(min_k, model, tokenizer, k=0.3)
        predictor.fit(df, labels)
        probs = predictor.predict_proba(df)
    """

    threshold_: float | None = None
    _calibrator: LogisticRegression | None = None

    def __init__(self, attack_fn: Callable, model, tokenizer,
                 text_col: str = "text", **attack_kwargs):
        self.attack_fn = attack_fn
        self.model = model
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.attack_kwargs = attack_kwargs

    @property
    def feature_key(self) -> str:
        return f"mia_{self.attack_fn.__name__}"

    def score(self, df: pd.DataFrame, *, cache_path=None) -> np.ndarray:
        """Raw MIA score per item. Higher = more likely memorized.

        Args:
            cache_path: If given, load scores from this .npy file if it exists,
                otherwise compute and save to it.
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                return np.load(cache_path)

        scores = np.array([
            self.attack_fn(self.model, self.tokenizer, text, **self.attack_kwargs)
            for text in tqdm(df[self.text_col], desc=self.attack_fn.__name__)
        ])

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(cache_path, scores)

        return scores

    def fit(self, df, labels, *, scores=None, **kwargs):
        """Find optimal threshold (maximize F1) and fit Platt calibrator.

        Args:
            scores: Pre-computed score array. If None, calls self.score(df).
        """
        if scores is None:
            scores = self.score(df)

        precision, recall, thresholds = precision_recall_curve(labels, scores)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                (precision + recall) > 0,
                2 * precision * recall / (precision + recall),
                0.0,
            )
        best_idx = np.argmax(f1[:-1])
        self.threshold_ = thresholds[best_idx]

        self._calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
        self._calibrator.fit(scores.reshape(-1, 1), labels)

    def predict(self, df, *, scores=None, **kwargs):
        """Binary prediction using fitted threshold.

        Args:
            scores: Pre-computed score array. If None, calls self.score(df).
        """
        if self.threshold_ is None:
            raise RuntimeError("Call fit() first")
        if scores is None:
            scores = self.score(df)
        return (scores >= self.threshold_).astype(int)

    def predict_proba(self, df, *, scores=None, **kwargs):
        """Calibrated P(contaminated) via Platt scaling.

        Args:
            scores: Pre-computed score array. If None, calls self.score(df).
        """
        if self._calibrator is None:
            raise RuntimeError("Call fit() first")
        if scores is None:
            scores = self.score(df)
        return self._calibrator.predict_proba(scores.reshape(-1, 1))

    def evaluate(self, df: pd.DataFrame, labels: np.ndarray,
                 *, scores=None) -> dict:
        """Compute metrics at best threshold and AUROC.

        Args:
            scores: Pre-computed score array. If None, calls self.score(df).
        """
        if self.threshold_ is None:
            raise RuntimeError("Call fit() first")
        if scores is None:
            scores = self.score(df)
        preds = (scores >= self.threshold_).astype(int)
        return {
            "threshold": self.threshold_,
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, zero_division=0),
            "recall": recall_score(labels, preds, zero_division=0),
            "f1": f1_score(labels, preds, zero_division=0),
            "auroc": roc_auc_score(labels, scores),
            "n_pos": int(labels.sum()),
            "n_neg": int((1 - labels).sum()),
        }


class PrecomputedPredictor(AttackPredictor):
    """Predictor that reads pre-computed score columns from a DataFrame.

    Reads columns named ``{col_prefix}{option}_{model_key}`` for each option,
    then takes the max across options as the MIA score. No model needed.

    Example::

        predictor = PrecomputedPredictor("hubble-1b", ["A", "B"], col_prefix="logprob_")
        predictor.fit(df, labels)
    """

    def __init__(self, model_key: str, option_cols: list[str],
                 col_prefix: str = "logprob_"):
        self.model_key = model_key
        self.option_cols = option_cols
        self.col_prefix = col_prefix

    @property
    def feature_key(self) -> str:
        return f"mia_precomputed_{self.model_key}"

    def score(self, df: pd.DataFrame, **kwargs) -> np.ndarray:
        lp_arr = np.column_stack([
            df[f"{self.col_prefix}{col}_{self.model_key}"].values
            for col in self.option_cols
        ])
        return lp_arr.max(axis=1)


# Backward-compatible alias
LogprobCorrectPredictor = PrecomputedPredictor
