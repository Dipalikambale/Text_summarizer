import unittest

import torch

from dialogue_summarizer.config import DataConfig
from dialogue_summarizer.data import tokenize_record
from dialogue_summarizer.inference import Summarizer
from dialogue_summarizer.preprocessing import normalize_record


class FakeInferenceTokenizer:
    # Let tests control the decoded text without loading a real tokenizer.
    decoded_text = "999"

    def num_special_tokens_to_add(self, pair: bool = False) -> int:
        # The fake vocabulary has no boundary tokens, simplifying chunk-size assertions.
        return 0

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        verbose: bool = False,
    ) -> list[int]:
        # Treat whitespace-separated numbers as token IDs for the chunking test.
        return [int(token) for token in text.split()]

    def prepare_for_model(
        self,
        token_ids: list[int],
        *,
        add_special_tokens: bool,
        return_tensors: None,
    ) -> dict[str, list[int]]:
        # Mimic the fields returned by Hugging Face prepare_for_model.
        return {"input_ids": token_ids}

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self.decoded_text


class FakeInferenceModel:
    def __init__(self) -> None:
        # Remember input lengths so the test can verify no chunk exceeds the limit.
        self.input_lengths: list[int] = []

    def to(self, device: torch.device) -> "FakeInferenceModel":
        return self

    def eval(self) -> "FakeInferenceModel":
        return self

    def generate(self, input_ids: torch.Tensor, **kwargs: object) -> list[list[int]]:
        # Record the current chunk size and return a short stand-in summary token.
        self.input_lengths.append(len(input_ids[0]))
        return [[999]]


class FakeTokenizer:
    def __call__(
        self,
        texts: list[str] | None = None,
        *,
        text_target: list[str] | None = None,
        max_length: int,
        truncation: bool,
    ) -> dict[str, list[list[int]]]:
        # Support both source-text tokenization and summary-target tokenization.
        values = text_target if text_target is not None else texts
        assert values is not None
        return {
            "input_ids": [
                [len(text), max_length, int(truncation)] for text in values
            ]
        }


class NormalizeRecordTests(unittest.TestCase):
    def test_trims_dialogue_and_summary(self) -> None:
        # Surrounding whitespace should be removed without changing the example text.
        self.assertEqual(
            normalize_record(
                {"dialogue": "  A: Hello.  ", "summary": "  They greet.  "}
            ),
            {"dialogue": "A: Hello.", "summary": "They greet."},
        )

    def test_rejects_missing_or_empty_text(self) -> None:
        # Each bad row should identify which required field needs correction.
        invalid_records = [
            ({"dialogue": "", "summary": "Summary"}, "dialogue"),
            ({"dialogue": "Dialogue", "summary": "  "}, "summary"),
            ({"dialogue": None, "summary": "Summary"}, "dialogue"),
        ]
        for record, field in invalid_records:
            with self.subTest(field=field, record=record):
                with self.assertRaisesRegex(ValueError, field):
                    normalize_record(record)

    def test_tokenize_record_encodes_inputs_and_target_labels(self) -> None:
        # The fake tokenizer records lengths to check the input/target settings.
        result = tokenize_record(
            {"dialogue": [" A: Hi. "], "summary": [" They greet. "]},
            FakeTokenizer(),
            DataConfig(max_input_length=32, max_target_length=16),
        )
        self.assertEqual(result["input_ids"], [[6, 32, 1]])
        self.assertEqual(result["labels"], [[11, 16, 1]])

    def test_long_dialogue_is_summarized_in_bounded_chunks(self) -> None:
        # Build an inference object with fakes so the test needs no model download.
        summarizer = Summarizer.__new__(Summarizer)
        summarizer.tokenizer = FakeInferenceTokenizer()
        summarizer.model = FakeInferenceModel()
        summarizer.device = torch.device("cpu")
        summarizer.max_input_length = 5
        summarizer.max_new_tokens = 2
        summarizer.num_beams = 1
        summarizer.chunk_overlap = 1

        # Provide more tokens than one chunk can hold to exercise hierarchical chunking.
        summary = summarizer.summarize("1 2 3 4 5 6 7 8 9 10 11 12")

        # Confirm multiple generations happened and every individual input fit.
        self.assertEqual(summary, "999")
        self.assertGreater(len(summarizer.model.input_lengths), 1)
        self.assertLessEqual(max(summarizer.model.input_lengths), 5)

    def test_summary_removes_tokenizer_newline_markers(self) -> None:
        # Simulate the visible <n> marker returned by Pegasus tokenizers.
        summarizer = Summarizer.__new__(Summarizer)
        summarizer.tokenizer = FakeInferenceTokenizer()
        summarizer.tokenizer.decoded_text = (
            "Patient reports fatigue.<n>Patient reports a headache."
        )
        summarizer.model = FakeInferenceModel()
        summarizer.device = torch.device("cpu")
        summarizer.max_new_tokens = 16
        summarizer.num_beams = 1

        # Generate once to verify the marker is cleaned before returning to the API.
        summary = summarizer._generate_summary([1, 2, 3])

        self.assertEqual(summary, "Patient reports fatigue. Patient reports a headache.")


if __name__ == "__main__":
    unittest.main()
