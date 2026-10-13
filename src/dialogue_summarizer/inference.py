"""Load a trained model and summarize dialogue text."""

import argparse
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import GenerationConfig, TrainingConfig


class Summarizer:
    """Reusable Transformer-backed dialogue summarizer."""

    def __init__(
        self,
        model_path: str = TrainingConfig.output_dir,
        max_input_length: int = GenerationConfig.max_input_length,
        max_new_tokens: int = GenerationConfig.max_new_tokens,
        num_beams: int = GenerationConfig.num_beams,
        chunk_overlap: int = 32,
    ) -> None:
        # Validate generation controls before loading potentially large model files.
        if (
            max_input_length <= 0
            or max_new_tokens <= 0
            or num_beams <= 0
            or chunk_overlap < 0
        ):
            raise ValueError("Generation limits and beam count must be positive.")
        # Load the vocabulary and trained sequence-to-sequence model.
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        # Use a GPU when available, otherwise run the model on the CPU.
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        # Disable training-only behavior such as dropout during summary generation.
        self.model.eval()
        # Store the limits used to divide and generate dialogue summaries.
        self.max_input_length = max_input_length
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        self.chunk_overlap = chunk_overlap

    def summarize(self, dialogue: str) -> str:
        """Summarize dialogue in token-bounded chunks without truncating input."""
        # Reject missing or whitespace-only text rather than returning an empty summary.
        if not isinstance(dialogue, str) or not dialogue.strip():
            raise ValueError("Dialogue must be a non-empty string.")

        # Reserve positions for tokens (such as start/end markers) added by the tokenizer.
        capacity = self.max_input_length - self.tokenizer.num_special_tokens_to_add(
            pair=False
        )
        if capacity <= 0:
            raise ValueError(
                "max_input_length must leave room for the tokenizer's special tokens."
            )

        # Convert the full dialogue to token IDs without truncating its ending.
        pending = self._encode(dialogue.strip())
        # Keep a small tail from each chunk so context is not lost at chunk boundaries.
        overlap = min(self.chunk_overlap, max(0, capacity - 1))
        with torch.inference_mode():
            while len(pending) > capacity:
                # Advance by capacity minus overlap so adjacent chunks share context.
                stride = capacity - overlap
                summaries = []
                for start in range(0, len(pending), stride):
                    # Slice one model-sized part of the original dialogue or intermediate summaries.
                    chunk = pending[start : start + capacity]
                    summaries.append(self._generate_summary(chunk))
                # Combine partial summaries into text and check if another reduction pass is needed.
                reduced = self._encode("\n".join(summaries))
                if len(reduced) >= len(pending):
                    # Stop instead of looping forever if the model did not shorten the input.
                    raise RuntimeError(
                        "The dialogue is too long to summarize without losing content; "
                        "the model did not reduce the chunk summaries."
                    )
                # Feed the shorter summary text through another pass until it fits the model.
                pending = reduced

            # Generate the final summary once the remaining text fits in one model input.
            return self._generate_summary(pending)

    def _encode(self, text: str) -> list[int]:
        # Add no special tokens here; they are inserted after chunking in _generate_summary.
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            verbose=False,
        )

    def _generate_summary(self, token_ids: list[int]) -> str:
        # Add model boundary tokens to the already-bounded input token IDs.
        prepared = self.tokenizer.prepare_for_model(
            token_ids,
            add_special_tokens=True,
            return_tensors=None,
        )
        # Add a batch dimension and move each tokenizer output tensor to the model device.
        inputs = {
            name: torch.tensor([values], device=self.device)
            for name, values in prepared.items()
        }
        # Generate candidate summaries using beam search.
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            num_beams=self.num_beams,
            early_stopping=True,
        )
        # Remove model-only tokens and normalize Pegasus' visible newline marker.
        decoded = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        return " ".join(decoded.replace("<n>", " ").split())


def main(argv: Sequence[str] | None = None) -> None:
    # Require the caller to provide either inline dialogue text or a file, but not both.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=TrainingConfig.output_dir)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Dialogue text to summarize.")
    input_group.add_argument(
        "--input-file", type=Path, help="UTF-8 file containing dialogue text."
    )
    # Parse the selected checkpoint and source of dialogue text.
    args = parser.parse_args(argv)
    # Read UTF-8 file content when file input was selected; otherwise use inline text.
    dialogue = args.text if args.text is not None else args.input_file.read_text(
        encoding="utf-8"
    )
    # Load the model and print the generated summary to the terminal.
    print(Summarizer(args.model_path).summarize(dialogue))


if __name__ == "__main__":
    main()
