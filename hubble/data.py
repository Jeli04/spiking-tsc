"""Data loading for Hubble perturbation datasets."""

import json
import pandas as pd
from datasets import load_dataset


def load_winogrande_perturbations(format: str = "infill") -> pd.DataFrame:
    """Load Hubble's WinoGrande perturbation dataset.

    Args:
        format: "infill" or "mcq"

    Returns:
        DataFrame with columns: orig_idx, sentence, option1, option2, answer,
        duplicates, paired_orig_idx, text, split
    """
    ds_name = f"allegrolab/testset_winogrande-{format}"
    ds = load_dataset(ds_name)

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            paired = json.loads(meta["paired_example"]) if meta["paired_example"] else None
            rows.append({
                "orig_idx": meta["orig_idx"],
                "sentence": meta["sentence"],
                "option1": meta["option1"],
                "option2": meta["option2"],
                "answer": int(meta["answer"]),
                "duplicates": example["duplicates"],
                "paired_orig_idx": paired["orig_idx"] if paired else None,
                "text": example["text"],
                "split": split_name,
                "format": format,
            })

    return pd.DataFrame(rows)


def load_mmlu_perturbations() -> pd.DataFrame:
    """Load Hubble's MMLU perturbation dataset.

    Returns:
        DataFrame with columns: orig_idx, question, choices, answer, subject,
        duplicates, text, split.
        answer is 0-indexed (0=A, 1=B, 2=C, 3=D).
        choices is a list of 4 strings.
    """
    ds = load_dataset("allegrolab/testset_mmlu")

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            rows.append({
                "orig_idx": meta["orig_idx"],
                "question": meta["question"],
                "choices": meta["choices"],  # list of 4 strings
                "answer": int(meta["answer"]),  # 0-indexed
                "subject": meta["subject"],
                "duplicates": example["duplicates"],
                "text": example["text"],
                "split": split_name,
            })

    return pd.DataFrame(rows)


def load_piqa_perturbations() -> pd.DataFrame:
    """Load Hubble's PIQA perturbation dataset.

    Returns:
        DataFrame with columns: orig_idx, goal, sol1, sol2, answer (0-indexed),
        duplicates, text, split.
    """
    ds = load_dataset("allegrolab/testset_piqa")

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            rows.append({
                "orig_idx": meta["orig_idx"],
                "goal": meta["goal"],
                "sol1": meta["sol1"],
                "sol2": meta["sol2"],
                "answer": int(meta["label"]),  # 0-indexed
                "duplicates": example["duplicates"],
                "text": example["text"],
                "split": split_name,
            })

    return pd.DataFrame(rows)


def load_popqa_perturbations() -> pd.DataFrame:
    """Load Hubble's PopQA perturbation dataset.

    Returns:
        DataFrame with columns: orig_idx, question, possible_answers (list of str),
        s_pop, o_pop, prop, duplicates, text, split.
    """
    ds = load_dataset(
        "allegrolab/testset_popqa",
        data_files={"validation": "testset_popqa_nodup.jsonl.gz"},
    )

    rows = []
    for example in ds["validation"]:
        meta = json.loads(example["meta"])
        rows.append({
            "orig_idx": meta["orig_idx"],
            "question": meta["question"],
            "possible_answers": json.loads(meta["possible_answers"]),  # list of str
            "s_pop": meta["s_pop"],
            "o_pop": meta["o_pop"],
            "prop": meta["prop"],
            "duplicates": meta["duplicates"],
            "text": example["text"],
            "split": "validation",
        })

    return pd.DataFrame(rows)


def load_hellaswag_perturbations() -> pd.DataFrame:
    """Load Hubble's HellaSwag perturbation dataset.

    Returns:
        DataFrame with columns: orig_idx, ctx, activity_label, endings (list of 4),
        answer (0-indexed), duplicates, text, split.
    """
    ds = load_dataset("allegrolab/testset_hellaswag")

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            rows.append({
                "orig_idx": meta["orig_idx"],
                "ctx": meta["ctx"],
                "ctx_a": meta["ctx_a"],
                "ctx_b": meta["ctx_b"],
                "activity_label": meta["activity_label"],
                "endings": meta["endings"],  # list of 4 strings
                "answer": int(meta["label"]),  # 0-indexed
                "duplicates": example["duplicates"],
                "text": example["text"],
                "split": split_name,
            })

    return pd.DataFrame(rows)


# Mapping from benchmark name to the DataFrame column containing the
# question/prompt only (without the answer).  Each benchmark stores
# its question in a different meta field.
QUESTION_COLUMNS = {
    'mmlu': 'question',
    'winogrande_infill': 'sentence',
    'winogrande_mcq': 'sentence',
    'piqa': 'goal',
    'hellaswag': 'ctx',
    'popqa': 'question',
}


def load_wikipedia_passages() -> pd.DataFrame:
    """Load Hubble's Wikipedia passages dataset.

    Returns:
        DataFrame with columns: title, duplicates, text, split.
    """
    ds = load_dataset("allegrolab/passages_wikipedia")

    rows = []
    for split_name in ds:
        for example in ds[split_name]:
            meta = json.loads(example["meta"])
            rows.append({
                "title": meta["title"],
                "duplicates": example["duplicates"],
                "text": example["text"],
                "split": split_name,
            })

    return pd.DataFrame(rows)


# Benchmark registry: name -> callable returning the full DataFrame.
# Use this instead of redefining per-script LOADERS dicts.
BENCHMARK_LOADERS = {
    'winogrande_infill': lambda: load_winogrande_perturbations('infill'),
    'winogrande_mcq': lambda: load_winogrande_perturbations('mcq'),
    'winogrande': lambda: pd.concat([
        load_winogrande_perturbations('infill'),
        load_winogrande_perturbations('mcq'),
    ], ignore_index=True),
    'mmlu': load_mmlu_perturbations,
    'piqa': load_piqa_perturbations,
    'popqa': load_popqa_perturbations,
    'hellaswag': load_hellaswag_perturbations,
    'wikipedia': load_wikipedia_passages,
}
