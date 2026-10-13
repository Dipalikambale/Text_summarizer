"""Dependency-free validation and normalization of dialogue records."""

from collections.abc import Mapping
from typing import Any


def normalize_record(record: Mapping[str, Any]) -> dict[str, str]:
    """Validate one SAMSum-shaped row and return normalized dialogue text."""
    # Read the two fields required by the summarization task.
    dialogue = record.get("dialogue")
    summary = record.get("summary")
    # Reject missing, incorrectly typed, or whitespace-only conversation text.
    if not isinstance(dialogue, str) or not dialogue.strip():
        raise ValueError("Each record must contain a non-empty string 'dialogue'.")
    # Reject unusable reference summaries for the same reasons.
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("Each record must contain a non-empty string 'summary'.")
    # Remove surrounding whitespace while keeping the actual content unchanged.
    return {"dialogue": dialogue.strip(), "summary": summary.strip()}
