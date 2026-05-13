"""Correctness predictors: estimate P(correct | x) from scalar confidence.

Three predictor types:
  - ConfidencePredictor: reads pre-computed confidence from a DataFrame column,
    applies Platt scaling. No model or tokenizer required.
  - LLMConfidencePredictor: evaluates an external LLM on benchmark examples to
    obtain per-example confidence, then applies Platt scaling. Two-phase
    (extract on GPU, fit/predict on CPU), following the same pattern as
    HiddenStatePredictor in mem_predictors.
  - RoBERTaCorrectnessPredictor: fine-tunes RoBERTa on benchmark text to predict
    correctness. Three-phase: train (GPU), extract scores (GPU), Platt scaling
    (CPU). Saves finetuned model for reuse.

NOTE: [pedagogical] Using an external LLM as a difficulty proxy works because
intrinsic difficulty is largely model-agnostic — questions that are hard for
one LLM tend to be hard for another. The external LLM should NOT have been
trained on the perturbation data, so its confidence reflects genuine difficulty
rather than memorization.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hubble.mem_predictors import Predictor


class ConfidencePredictor(Predictor):
    """Predictor that reads scalar confidence and applies Platt scaling."""

    def __init__(self) -> None:
        self._lr: LogisticRegression | None = None

    @property
    def feature_key(self) -> str:
        return 'confidence'

    def fit(self, df, labels, **kwargs):
        c = df['confidence'].values.reshape(-1, 1)
        self._lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._lr.fit(c, labels)

    def predict(self, df, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(df['confidence'].values)
        return (p >= 0.5).astype(int)

    def predict_proba(self, df, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(df['confidence'].values)
        return np.column_stack([1 - p, p])

    def _calibrated_proba(self, confidence: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError('Call fit() first')
        return self._lr.predict_proba(confidence.reshape(-1, 1))[:, 1]


# ==========================================================================
# LLM confidence predictor
# ==========================================================================

class LLMConfidencePredictor(Predictor):
    """Correctness predictor that evaluates an external LLM, then applies Platt scaling.

    Two-phase usage (mirrors HiddenStatePredictor):
      1. extract_confidence() — GPU. Evaluate the LLM on benchmark examples,
         cache per-example confidence scores to a parquet file.
      2. fit() / predict_proba() — CPU. Platt scaling on pre-extracted
         confidence arrays.

    NOTE: [design] For MC benchmarks, confidence = softmax probability of the
    correct answer. For generative benchmarks (popqa), confidence = max mean
    log-prob over possible answers.

    Usage::

        predictor = LLMConfidencePredictor('meta-llama/Llama-3.1-8B')

        # Phase 1: extract confidence (GPU)
        conf = predictor.extract_confidence(model, tokenizer, df, 'mmlu',
                                        cache_path='cache/llama_mmlu.parquet')

        # Phase 2: fit + predict (CPU, on pre-extracted arrays)
        predictor.fit(conf[train_idx], labels_train)
        c_hat = predictor.predict_proba(conf[test_idx])
    """

    # Benchmark -> evaluation type
    _MC_BENCHMARKS = {'mmlu', 'hellaswag', 'winogrande', 'piqa'}
    _GEN_BENCHMARKS = {'popqa'}

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        self._label = model_id.split('/')[-1]
        self._lr: LogisticRegression | None = None

    @property
    def feature_key(self) -> str:
        return f'llm_confidence_{self._label}'

    # ------------------------------------------------------------------
    # Phase 1: Extract confidence (GPU)
    # ------------------------------------------------------------------

    def extract_confidence(
        self,
        model,
        tokenizer,
        df: pd.DataFrame,
        benchmark: str,
        cache_path: Path | str | None = None,
    ) -> np.ndarray:
        """Evaluate the LLM on benchmark examples and return confidence scores.

        Args:
            model: Pre-loaded causal LM (on GPU).
            tokenizer: Corresponding tokenizer.
            df: DataFrame with benchmark examples (as returned by hubble.data loaders).
            benchmark: One of 'mmlu', 'hellaswag', 'winogrande', 'piqa', 'popqa'.
            cache_path: If provided, cache results as parquet. Loads from cache if exists.

        Returns:
            1D numpy array of confidence scores, aligned with df rows.
            MC benchmarks: softmax probability of the correct answer.
            Gen benchmarks: max mean log-prob over possible answers.
        """
        from hubble.eval import evaluate_benchmark_df

        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                cached = pd.read_parquet(cache_path)
                print(f'[{self.feature_key}] Loaded cached confidence ({len(cached)} rows)')
                return cached['confidence'].values

        label = self._label

        if benchmark in self._MC_BENCHMARKS or benchmark in self._GEN_BENCHMARKS:
            result_df = evaluate_benchmark_df(model, tokenizer, df, label, benchmark)
            confidence = result_df[f'confidence_{label}'].values
        else:
            raise ValueError(f'Unknown benchmark: {benchmark}')

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_df = pd.DataFrame({
                'orig_idx': df['orig_idx'].values,
                'confidence': confidence,
            })
            if f'acc_{label}' in result_df.columns:
                cache_df['acc'] = result_df[f'acc_{label}'].values
            cache_df.to_parquet(cache_path, index=False)
            print(f'[{self.feature_key}] Cached confidence -> {cache_path}')

        return confidence

    # ------------------------------------------------------------------
    # Phase 2: Fit / predict (CPU, on pre-extracted confidence arrays)
    # ------------------------------------------------------------------

    def fit(self, data, labels, **kwargs) -> None:
        """Fit Platt scaling on confidence array.

        Args:
            data: 1D numpy array of confidence scores, or DataFrame with 'confidence' column.
            labels: Binary correctness labels.
        """
        confidence = self._get_confidence(data)
        self._lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._lr.fit(confidence.reshape(-1, 1), labels)

    def predict(self, data, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(self._get_confidence(data))
        return (p >= 0.5).astype(int)

    def predict_proba(self, data, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(self._get_confidence(data))
        return np.column_stack([1 - p, p])

    def _get_confidence(self, data) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            return data['confidence'].values
        return np.asarray(data)

    def _calibrated_proba(self, confidence: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError('Call fit() first')
        return self._lr.predict_proba(confidence.reshape(-1, 1))[:, 1]


# ==========================================================================
# RoBERTa correctness predictor
# ==========================================================================

class RoBERTaCorrectnessPredictor(Predictor):
    """Fine-tuned RoBERTa predictor for predicting LLM correctness from text.

    Three-phase usage:
      1. train_model() — GPU. Fine-tune RoBERTa on (text, correctness_label)
         pairs. Saves finetuned model + tokenizer to model_dir.
      2. extract_scores() — GPU. Load saved model, run inference on texts,
         return sigmoid(logit) scores. Caches to .npz file.
      3. fit() / predict_proba() — CPU. Platt scaling on pre-extracted scores.

    NOTE: [design] The RoBERTa model learns text features that predict
    difficulty (e.g., question complexity, topic), going beyond a scalar
    confidence feature. It should be trained only on clean items to avoid
    learning contamination artifacts.

    Usage::

        predictor = RoBERTaCorrectnessPredictor(model_dir='cache/roberta_correctness')

        # Phase 1: train (GPU) — skip if model_dir already exists
        predictor.train_model(train_texts, train_labels)

        # Phase 2: extract scores (GPU)
        scores = predictor.extract_scores(texts, cache_path='cache/roberta_scores.npz')

        # Phase 3: fit + predict (CPU, on pre-extracted scores)
        predictor.fit(scores[cal_idx], labels_cal)
        c_hat = predictor.predict_proba(scores[sim_idx])
    """

    def __init__(
        self,
        model_dir: str | Path,
        base_model: str = 'roberta-base',
        max_length: int = 512,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.base_model = base_model
        self.max_length = max_length
        self._lr: LogisticRegression | None = None

    @property
    def feature_key(self) -> str:
        return f'roberta_correctness'

    # ------------------------------------------------------------------
    # Phase 1: Train model (GPU)
    # ------------------------------------------------------------------

    def train_model(
        self,
        texts: list[str],
        labels: list[int] | np.ndarray,
        epochs: int = 3,
        batch_size: int = 32,
        lr: float = 2e-5,
        seed: int = 42,
        eval_texts: list[str] | None = None,
        eval_labels: list[int] | np.ndarray | None = None,
        classifier_dropout: float | None = None,
        label_smoothing: float = 0.0,
        freeze_layers: tuple[int, int] | None = None,
    ) -> None:
        """Fine-tune RoBERTa on text → binary correctness. Saves to model_dir.

        Skips training if model_dir already exists (cached).

        Args:
            texts: Training texts.
            labels: Binary correctness labels (0/1).
            epochs: Number of training epochs.
            batch_size: Training batch size.
            lr: Learning rate.
            seed: Random seed.
            eval_texts: Optional eval texts for per-epoch metrics.
            eval_labels: Optional eval labels.
            classifier_dropout: Dropout for classification head (None = HF default).
            label_smoothing: Label smoothing factor (0 = none, e.g. 0.1 maps 0/1 to 0.05/0.95).
            freeze_layers: Tuple (start, end) to freeze encoder layers [start, end).
        """
        if self.model_dir.exists() and (self.model_dir / 'config.json').exists():
            print(f'[{self.feature_key}] Model already saved at {self.model_dir}, skipping training')
            return

        import torch
        import torch.nn as nn
        from transformers import (
            RobertaForSequenceClassification,
            RobertaTokenizer,
            Trainer,
            TrainingArguments,
        )

        import random

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        tokenizer = RobertaTokenizer.from_pretrained(self.base_model)
        model_kwargs = {'num_labels': 1}
        if classifier_dropout is not None:
            model_kwargs['classifier_dropout'] = classifier_dropout
        model = RobertaForSequenceClassification.from_pretrained(
            self.base_model, **model_kwargs,
        )

        # Freeze encoder layers if requested
        if freeze_layers is not None:
            start, end = freeze_layers
            encoder_layers = model.roberta.encoder.layer
            if end > len(encoder_layers):
                raise ValueError(
                    f'freeze_layers end={end} exceeds number of layers ({len(encoder_layers)})')
            for i in range(start, end):
                for param in encoder_layers[i].parameters():
                    param.requires_grad = False
            print(f'[{self.feature_key}] Froze encoder layers [{start}, {end})')

        # Build eval set: use provided eval data, or hold out 15% of training data
        if eval_texts is not None and eval_labels is not None:
            train_ds = self._make_dataset(tokenizer, texts, labels)
            eval_ds = self._make_dataset(tokenizer, eval_texts, eval_labels)
        else:
            indices = list(range(len(texts)))
            random.shuffle(indices)
            n_eval = max(1, int(len(indices) * 0.15))
            eval_idx = indices[:n_eval]
            train_idx = indices[n_eval:]
            labels_list = list(labels)
            train_ds = self._make_dataset(
                tokenizer,
                [texts[i] for i in train_idx],
                [labels_list[i] for i in train_idx],
            )
            eval_ds = self._make_dataset(
                tokenizer,
                [texts[i] for i in eval_idx],
                [labels_list[i] for i in eval_idx],
            )
            print(f'[{self.feature_key}] Held out {n_eval} items for best-epoch selection')

        # Custom trainer with BCEWithLogitsLoss + optional label smoothing
        _ls = label_smoothing

        class _BCETrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                batch_labels = inputs.pop('labels')
                if _ls > 0:
                    batch_labels = batch_labels * (1 - _ls) + 0.5 * _ls
                outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1)
                loss = nn.BCEWithLogitsLoss()(logits, batch_labels)
                return (loss, outputs) if return_outputs else loss

        checkpoint_dir = str(self.model_dir / '_checkpoints')
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size * 2,
            learning_rate=lr,
            weight_decay=0.01,
            eval_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            logging_steps=50,
            seed=seed,
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            report_to='none',
            warmup_ratio=0.1,
        )

        trainer = _BCETrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
        )

        print(f'[{self.feature_key}] Training RoBERTa ({len(texts)} examples, {epochs} epochs)...')
        trainer.train()
        print(f'[{self.feature_key}] Best checkpoint: {trainer.state.best_model_checkpoint}')

        # Save finetuned model + tokenizer
        self.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.model_dir)
        tokenizer.save_pretrained(self.model_dir)
        print(f'[{self.feature_key}] Saved finetuned model -> {self.model_dir}')

    # ------------------------------------------------------------------
    # Phase 2: Extract scores (GPU)
    # ------------------------------------------------------------------

    def extract_scores(
        self,
        texts: list[str],
        batch_size: int = 64,
        cache_path: Path | str | None = None,
    ) -> np.ndarray:
        """Run inference with saved model, return sigmoid(logit) scores.

        Args:
            texts: Input texts.
            batch_size: Inference batch size.
            cache_path: If provided, cache scores to .npz. Loads if exists.

        Returns:
            1D numpy array of P(correct) scores in [0, 1].
        """
        if cache_path is not None:
            cache_path = Path(cache_path)
            if cache_path.exists():
                scores = np.load(cache_path)['scores']
                print(f'[{self.feature_key}] Loaded cached scores ({len(scores)} items)')
                return scores

        import torch
        from transformers import RobertaForSequenceClassification, RobertaTokenizer
        from tqdm import tqdm

        tokenizer = RobertaTokenizer.from_pretrained(str(self.model_dir))
        model = RobertaForSequenceClassification.from_pretrained(str(self.model_dir))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()

        all_scores = []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        for start in tqdm(range(0, len(texts), batch_size),
                          total=n_batches, desc='Extracting RoBERTa scores'):
            batch_texts = texts[start:start + batch_size]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors='pt',
            ).to(device)

            with torch.no_grad():
                logits = model(**enc).logits.squeeze(-1)
                probs = torch.sigmoid(logits).cpu().numpy()
            all_scores.append(probs)

        scores = np.concatenate(all_scores)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache_path, scores=scores)
            print(f'[{self.feature_key}] Cached scores -> {cache_path}')

        return scores

    # ------------------------------------------------------------------
    # Phase 3: Fit / predict (CPU, on pre-extracted scores)
    # ------------------------------------------------------------------

    def fit(self, data, labels, **kwargs) -> None:
        """Fit Platt scaling on pre-extracted score array."""
        scores = np.asarray(data).ravel()
        self._lr = LogisticRegression(solver='lbfgs', max_iter=1000)
        self._lr.fit(scores.reshape(-1, 1), labels)

    def predict(self, data, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(np.asarray(data).ravel())
        return (p >= 0.5).astype(int)

    def predict_proba(self, data, **kwargs) -> np.ndarray:
        p = self._calibrated_proba(np.asarray(data).ravel())
        return np.column_stack([1 - p, p])

    def _calibrated_proba(self, scores: np.ndarray) -> np.ndarray:
        if self._lr is None:
            raise RuntimeError('Call fit() first')
        return self._lr.predict_proba(scores.reshape(-1, 1))[:, 1]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_dataset(self, tokenizer, texts, labels):
        """Create a simple map-style dataset for HF Trainer."""
        import torch
        from torch.utils.data import Dataset as TorchDataset

        enc = tokenizer(
            list(texts), padding=True, truncation=True,
            max_length=self.max_length, return_tensors='pt',
        )
        label_tensor = torch.tensor(list(labels), dtype=torch.float32)

        class _DS(TorchDataset):
            def __len__(self):
                return len(label_tensor)
            def __getitem__(self, idx):
                return {
                    'input_ids': enc['input_ids'][idx],
                    'attention_mask': enc['attention_mask'][idx],
                    'labels': label_tensor[idx],
                }

        return _DS()


# ==========================================================================
# Registry
# ==========================================================================

CORRECTNESS_PREDICTORS: dict[str, type[Predictor]] = {
    'platt': ConfidencePredictor,
}
