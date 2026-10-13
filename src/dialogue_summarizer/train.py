"""Fine-tune a pretrained sequence-to-sequence model on SAMSum."""

import argparse
from typing import Sequence

import torch
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


def train(
    model_name: str = TrainingConfig.model_name,
    output_dir: str = TrainingConfig.output_dir,
    epochs: int = TrainingConfig.num_train_epochs,
    batch_size: int = TrainingConfig.per_device_batch_size,
    learning_rate: float = TrainingConfig.learning_rate,
    max_train_samples: int | None = None,
    max_eval_samples: int | None = None,
) -> None:
    """Run the complete dataset-to-checkpoint fine-tuning pipeline."""
    # Load the pretrained tokenizer and model weights from the model repository.
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    # Fetch SAMSum locally when available or download it from Hugging Face.
    dataset = load_samsum()
    if max_train_samples is not None:
        # Restrict training rows when the user requests a quick smoke run.
        dataset["train"] = dataset["train"].select(
            range(min(max_train_samples, len(dataset["train"])))
        )
    if max_eval_samples is not None:
        # Restrict validation rows to shorten the smoke-run evaluation.
        dataset["validation"] = dataset["validation"].select(
            range(min(max_eval_samples, len(dataset["validation"])))
        )

    # Convert dialogues and summaries into separate model-ready token sequences.
    tokenized_train = dataset["train"].map(
        lambda batch: tokenize_record(batch, tokenizer, DataConfig()),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing SAMSum training dialogues",
    )
    tokenized_validation = dataset["validation"].map(
        lambda batch: tokenize_record(batch, tokenizer, DataConfig()),
        batched=True,
        remove_columns=dataset["validation"].column_names,
        desc="Tokenizing SAMSum validation dialogues",
    )
    # Prepare a local ROUGE scorer so training does not fetch a metric script online.
    metric = load_rouge()
    # Configure optimization, checkpointing, evaluation, and device precision.
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=TrainingConfig.weight_decay,
        warmup_ratio=TrainingConfig.warmup_ratio,
        num_train_epochs=epochs,
        predict_with_generate=True,
        generation_max_length=DataConfig.max_target_length,
        fp16=torch.cuda.is_available(),
        logging_steps=TrainingConfig.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="rougeL",
        greater_is_better=True,
        report_to="none",
        seed=TrainingConfig.seed,
    )
    # Seq2SeqTrainer handles batching, gradients, periodic evaluation, and saving.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model),
        processing_class=tokenizer,
        compute_metrics=lambda prediction: compute_rouge(
            prediction, tokenizer, metric
        ),
    )
    # Update the pretrained weights using the SAMSum training dialogues.
    trainer.train()
    # Save the final model and its tokenizer together for later inference.
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Fine-tuned model and tokenizer saved to: {output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    # Define command-line options so training parameters can be changed without editing code.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=TrainingConfig.model_name)
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--epochs", type=int, default=TrainingConfig.num_train_epochs)
    parser.add_argument(
        "--batch-size", type=int, default=TrainingConfig.per_device_batch_size
    )
    parser.add_argument(
        "--learning-rate", type=float, default=TrainingConfig.learning_rate
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    # Parse the user's command-line arguments.
    args = parser.parse_args(argv)
    # Fail early with a readable message if numeric training settings are invalid.
    if args.epochs <= 0 or args.batch_size <= 0 or args.learning_rate <= 0:
        parser.error("epochs, batch-size, and learning-rate must be positive.")
    if args.max_train_samples is not None and args.max_train_samples <= 0:
        parser.error("--max-train-samples must be positive.")
    if args.max_eval_samples is not None and args.max_eval_samples <= 0:
        parser.error("--max-eval-samples must be positive.")
    # Pass validated settings into the reusable training function.
    train(
        model_name=args.model_name,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )


if __name__ == "__main__":
    main()
