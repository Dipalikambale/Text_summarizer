"""SAMSum loading and tokenization helpers."""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import DataConfig
from .preprocessing import normalize_record


def load_samsum(dataset_name: str = DataConfig.dataset_name) -> Any:
    """Load local SAMSum parquet files or fetch the dataset from Hugging Face."""
    # Import here so lightweight validation code does not require datasets at import time.
    from datasets import load_dataset

    # Prefer downloaded parquet files when all three dataset splits are present.
    local_files = {
        split: Path("data") / f"{split}.parquet"
        for split in ("train", "validation", "test")
    }
    if all(path.is_file() for path in local_files.values()):
        # Load each local file under its expected split name.
        dataset = load_dataset(
            "parquet",
            data_files={split: str(path) for split, path in local_files.items()},
        )
    else:
        # Otherwise let Hugging Face Datasets download the named public dataset.
        dataset = load_dataset(dataset_name)
    # Check that each stage of training and evaluation has its required split.
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits.difference(dataset.keys())
    if missing_splits:
        raise ValueError(
            f"Dataset '{dataset_name}' is missing required splits: "
            f"{', '.join(sorted(missing_splits))}."
        )

    for split_name in required_splits:
        # Check every split has both model input text and its target summary.
        columns = set(dataset[split_name].column_names)
        missing_columns = {"dialogue", "summary"}.difference(columns)
        if missing_columns:
            raise ValueError(
                f"Dataset split '{split_name}' is missing required columns: "
                f"{', '.join(sorted(missing_columns))}."
            )
    return dataset


def tokenize_record(
    batch: Mapping[str, list[str]],
    tokenizer: Any,
    config: DataConfig = DataConfig(),
) -> dict[str, Any]:
    """Tokenize batched dialogues and target summaries for seq2seq training."""
    # Validate and trim each row before passing text to the tokenizer.
    rows = [normalize_record(row) for row in _batch_rows(batch)]
    dialogues = [row["dialogue"] for row in rows]
    summaries = [row["summary"] for row in rows]
    # Convert input dialogues to token IDs and truncate overlong examples.
    model_inputs = tokenizer(
        dialogues,
        max_length=config.max_input_length,
        truncation=True,
    )
    # Tokenize reference summaries separately; these become training labels.
    targets = tokenizer(
        text_target=summaries,
        max_length=config.max_target_length,
        truncation=True,
    )
    # The trainer compares generated output tokens against these target IDs.
    model_inputs["labels"] = targets["input_ids"]
    return model_inputs


def _batch_rows(batch: Mapping[str, list[str]]) -> list[dict[str, str]]:
    # Extract both columns and make sure this batched call has matching row counts.
    dialogues = batch.get("dialogue")
    summaries = batch.get("summary")
    if dialogues is None or summaries is None or len(dialogues) != len(summaries):
        raise ValueError("A batch must have equally sized 'dialogue' and 'summary' columns.")
    # Rebuild row-shaped records so each one can use the shared validator.
    return [
        {"dialogue": dialogue, "summary": summary}
        for dialogue, summary in zip(dialogues, summaries)
    ]
