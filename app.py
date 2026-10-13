"""Hugging Face Space UI for dialogue summarization."""

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr

# Make the src-layout package importable when the Space runs this root-level app.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dialogue_summarizer.inference import Summarizer

# Use a SAMSum-fine-tuned checkpoint instead of the eight-example local smoke model.
MODEL_ID = os.environ.get(
    "SUMMARIZER_MODEL",
    "philschmid/flan-t5-base-samsum",
)


@lru_cache(maxsize=1)
def get_summarizer() -> Summarizer:
    """Load the model once and reuse it for subsequent requests."""
    return Summarizer(MODEL_ID, max_new_tokens=96, num_beams=4)


def summarize_dialogue(dialogue: str) -> str:
    """Validate a request and generate its dialogue summary."""
    if not dialogue.strip():
        raise gr.Error("Enter a dialogue before requesting a summary.")
    if len(dialogue) > 50_000:
        raise gr.Error("Dialogue must be 50,000 characters or shorter.")
    return get_summarizer().summarize(dialogue)


# Build the browser interface shown on the Hugging Face Space page.
demo = gr.Interface(
    fn=summarize_dialogue,
    inputs=gr.Textbox(
        label="Dialogue",
        placeholder="Paste a speaker-labelled conversation here...",
        lines=14,
        max_lines=30,
    ),
    outputs=gr.Textbox(label="Summary", lines=5),
    title="Dialogue Summarizer",
    description=(
        "Summarize a conversation with a model fine-tuned on SAMSum. "
        "Long conversations are summarized in overlapping chunks."
    ),
    examples=[
        [
            "Doctor: What brings you here today?\n"
            "Patient: I've had a headache and felt unusually tired for three days.\n"
            "Doctor: Have you had a fever?\n"
            "Patient: No, but I haven't been sleeping well."
        ]
    ],
)

# Start the web server when the Hugging Face Space launches this file.
if __name__ == "__main__":
    demo.launch()
