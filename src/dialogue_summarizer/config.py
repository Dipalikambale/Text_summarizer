"""Shared defaults for model training and generation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    # Use the public SAMSum repository that provides dialogue and summary examples.
    dataset_name: str = "knkarthick/samsum"
    # Limit source dialogue length so batches fit into model memory.
    max_input_length: int = 512
    # Limit the reference summary length used for training.
    max_target_length: int = 128


@dataclass(frozen=True)
class TrainingConfig:
    # Start from a pretrained Pegasus summarization model.
    model_name: str = "google/pegasus-cnn_dailymail"
    # Save the trained weights and tokenizer in this project folder.
    output_dir: str = "models/pegasus-samsum"
    # Make three passes over the training data by default.
    num_train_epochs: int = 3
    # Number of examples processed together by each device.
    per_device_batch_size: int = 2
    # Controls how much model weights change at each optimizer update.
    learning_rate: float = 5e-5
    # Penalize overly large weight values to help limit overfitting.
    weight_decay: float = 0.01
    # Gradually warm up the learning rate during the first 10% of training.
    warmup_ratio: float = 0.1
    # Print training progress every 50 optimizer steps.
    logging_steps: int = 50
    # Fix random-number generation so experiments are more reproducible.
    seed: int = 42


@dataclass(frozen=True)
class GenerationConfig:
    # Keep inference input within the model's supported context length.
    max_input_length: int = 512
    # Stop generation after at most 128 new summary tokens.
    max_new_tokens: int = 128
    # Explore four candidate sequences and return the best-scoring one.
    num_beams: int = 4
