"""Evaluate a fine-tuned summarization model against SAMSum test references."""

import argparse
import json
import tempfile
from typing import Sequence

from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .config import DataConfig, TrainingConfig
from .data import load_samsum, tokenize_record
from .metrics import compute_rouge, load_rouge


def evaluate(
    model_path: str = TrainingConfig.output_dir,
    max_samples: int | None = None,
    batch_size: int = 2,
) -> dict[str, float]:
    """Generate summaries for the test split and return ROUGE scores."""
    # Load the requested checkpoint's model weights and matching tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    # Reuse the same SAMSum test examples when comparing different checkpoints.
    dataset = load_samsum()
    test_data = dataset["test"]
    if max_samples is not None:
        # Optionally shorten the evaluation run for a quick smoke test.
        test_data = test_data.select(range(min(max_samples, len(test_data))))
    # Tokenize the dialogue inputs and expected reference summaries.
    tokenized_test = test_data.map(
        lambda batch: tokenize_record(batch, tokenizer, DataConfig()),
        batched=True,
        remove_columns=test_data.column_names,
        desc="Tokenizing SAMSum test split",
    )
    # Create the ROUGE scorer locally instead of downloading a metric script.
    metric = load_rouge()
    # Keep temporary Trainer bookkeeping outside the user's model folder.
    with tempfile.TemporaryDirectory(prefix="dialogue-summarizer-evaluate-") as temp_dir:
        # Use Trainer's generation loop to predict summaries on the test split.
        trainer = Seq2SeqTrainer(
            model=model,
            args=Seq2SeqTrainingArguments(
                output_dir=temp_dir,
                per_device_eval_batch_size=batch_size,
                predict_with_generate=True,
                generation_max_length=DataConfig.max_target_length,
                report_to="none",
            ),
            # Pad variable-length input examples into batches acceptable to the model.
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
            processing_class=tokenizer,
            compute_metrics=lambda prediction: compute_rouge(
                prediction, tokenizer, metric
            ),
        )
        # Generate test predictions and request metrics with a recognizable prefix.
        prediction = trainer.predict(tokenized_test, metric_key_prefix="test")
    # Keep only ROUGE values and remove the Trainer's "test_" name prefix.
    scores = {
        name.removeprefix("test_"): float(score)
        for name, score in prediction.metrics.items()
        if name.startswith("test_rouge")
    }
    return scores


def compare_models(
    model_paths: Sequence[str],
    max_samples: int | None = None,
    batch_size: int = 2,
) -> dict[str, dict[str, float]]:
    """Evaluate each model on the same SAMSum test examples for comparison."""
    if not model_paths:
        # A comparison without any candidate models is a caller error.
        raise ValueError("Provide at least one model path to compare.")
    # Run the identical evaluation settings for each checkpoint.
    return {
        model_path: evaluate(model_path, max_samples, batch_size)
        for model_path in model_paths
    }


def main(argv: Sequence[str] | None = None) -> None:
    # Accept repeated --model-path flags so one command can compare checkpoints.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        action="append",
        help="Model checkpoint or Hugging Face model ID. Repeat to compare models.",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    # Convert command-line text into typed settings.
    args = parser.parse_args(argv)
    # Catch invalid sample limits before loading large checkpoints.
    if args.max_samples is not None and args.max_samples <= 0:
        parser.error("--max-samples must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    # Fall back to the configured fine-tuned model if no path was supplied.
    model_paths = args.model_path or [TrainingConfig.output_dir]
    # Print results as readable, machine-friendly JSON.
    print(
        json.dumps(
            compare_models(model_paths, args.max_samples, args.batch_size),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
