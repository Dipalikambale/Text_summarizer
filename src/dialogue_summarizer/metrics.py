"""ROUGE metric setup and normalization."""

from collections import defaultdict
from typing import Any


def load_rouge() -> Any:
    """Create a local ROUGE scorer without requiring a Hub metric script."""
    from rouge_score.rouge_scorer import RougeScorer

    # Score unigram, bigram, longest-common-subsequence, and summary-level overlap.
    return RougeScorer(["rouge1", "rouge2", "rougeL", "rougeLsum"], use_stemmer=True)


def compute_rouge(eval_prediction: Any, tokenizer: Any, metric: Any) -> dict[str, float]:
    """Decode generated token IDs and calculate the standard ROUGE scores."""
    # Trainer predictions may contain token IDs plus extra model outputs.
    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    # Replace Trainer's ignored-label marker with padding before decoding.
    labels[labels == -100] = tokenizer.pad_token_id
    # Turn model and reference token IDs back into readable strings.
    decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Store cumulative F1 scores for calculating a dataset-level mean.
    totals: dict[str, float] = defaultdict(float)
    for prediction, reference in zip(decoded_predictions, decoded_labels):
        # ROUGE expects the reference first and the generated text second.
        scores = metric.score(reference.strip(), prediction.strip())
        for name, score in scores.items():
            # Accumulate F1 rather than precision/recall for each metric.
            totals[name] += score.fmeasure
    count = len(decoded_predictions)
    if count == 0:
        # Return a complete metric shape when the input batch is empty.
        return {name: 0.0 for name in ("rouge1", "rouge2", "rougeL", "rougeLsum")}
    # Average each score over the examples in this prediction batch.
    return {name: score / count for name, score in totals.items()}
