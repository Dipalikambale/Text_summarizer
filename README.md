---
title: Dialogue Summarizer
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
---

# Dialogue Summarizer

A modular end-to-end dialogue summarization project using Hugging Face Transformers, PyTorch, and the SAMSum dataset. It follows the five-stage architecture in the project brief:

1. **Ingest** SAMSum dialogue and summary examples from Hugging Face Datasets.
2. **Transform** and tokenize records for sequence-to-sequence training.
3. **Fine-tune** a pretrained Pegasus model on SAMSum.
4. **Evaluate** generated summaries with ROUGE.
5. **Serve** summaries through a command-line interface or a FastAPI endpoint.

## Requirements and setup

Python 3.11 is recommended. A CUDA-enabled PyTorch installation is recommended for training; CPU execution is supported but fine-tuning will be slow. Install PyTorch using the instructions for your platform if the default package index does not provide the right build.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[api]"
```

The first training or evaluation run downloads the tokenizer and pretrained model from Hugging Face. SAMSum is fetched from Hugging Face Datasets when local parquet files are absent; if Hub dataset loading is unavailable, place `train.parquet`, `validation.parquet`, and `test.parquet` in `data/`.

## Fine-tune

```powershell
dialogue-train
```

This uses `knkarthick/samsum`, starts from `google/pegasus-cnn_dailymail`, and writes the final model and tokenizer to `models/pegasus-samsum`. You can override the main settings:

```powershell
dialogue-train --model-name google/pegasus-cnn_dailymail --output-dir models/pegasus-samsum --epochs 3 --batch-size 2 --learning-rate 5e-5
```

To run a quick smoke experiment, use `--max-train-samples 64 --max-eval-samples 32`. This limits examples; it does not replace a full training run.

### Fine-tune on a Colab GPU

For a faster full-data run than CPU training, upload [colab_finetune.ipynb](./colab_finetune.ipynb) to Google Colab, select **Runtime > Change runtime type > GPU**, and run its cells in order. The notebook uses the same Pegasus checkpoint and SAMSum dataset, saves checkpoints and the finished model to Google Drive, and resumes from the latest checkpoint after a disconnect. It requires access to a Google account; this project cannot start or access a Colab runtime on your behalf.

### Use an already fine-tuned model without training

To skip the training run, upload [colab_pretrained_samsum.ipynb](./colab_pretrained_samsum.ipynb) to Colab and run its cells. It loads `philschmid/flan-t5-base-samsum`, a public checkpoint already fine-tuned on SAMSum, and summarizes a dialogue directly. The first run downloads about 945 MB of model weights; no dataset preprocessing or Drive mount is needed.

### Deploy a public web app

The root-level `app.py`, `requirements.txt`, and this README's Hugging Face Spaces metadata are configured for a Gradio Space. Create a new public Gradio Space on Hugging Face, then push this repository to that Space's Git remote. The Space loads `philschmid/flan-t5-base-samsum` on first use; model weights are downloaded by the Space and are not stored in this Git repository.

## Evaluate

```powershell
dialogue-evaluate --model-path models/pegasus-samsum
```

Evaluation runs generation on the SAMSum test split and prints ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores. Pass `--max-samples 100` to evaluate a subset.

Compare checkpoints using the same test examples:

```powershell
dialogue-evaluate --model-path models/pegasus-samsum --model-path google/pegasus-cnn_dailymail
```

The Python API also exposes `compare_models([...])` from `dialogue_summarizer.evaluate`.

## Summarize a dialogue

```powershell
dialogue-summarize --model-path models/pegasus-samsum --text "A: Are we still meeting at 3? B: Yes, see you there."
```

Alternatively, provide a UTF-8 text file:

```powershell
dialogue-summarize --model-path models/pegasus-samsum --input-file dialogue.txt
```

## Run the API

The API loads the model lazily when the first request arrives. Set `SUMMARIZER_MODEL` to a local fine-tuned model directory or Hugging Face model ID (defaults to `models/pegasus-samsum`).

```powershell
$env:SUMMARIZER_MODEL = "models/pegasus-samsum"
uvicorn dialogue_summarizer.api:app --app-dir src --host 127.0.0.1 --port 8000
```

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/summarize `
  -ContentType "application/json" `
  -Body '{"dialogue":"A: Are we still meeting at 3? B: Yes, see you there."}'
```

The response is JSON in the form `{"summary": "..."}`. `GET /health` reports API liveness; it does not force model loading.

Long inputs up to the API's 50,000-character request limit are split into overlapping token-bounded chunks and summarized hierarchically. This prevents the tokenizer from silently dropping everything after the first 512 input tokens.

## Project layout

```text
src/dialogue_summarizer/
  api.py          FastAPI application and request schema
  config.py       Shared training and generation defaults
  data.py         SAMSum ingestion, schema checks, and tokenization
  evaluate.py     Test-split generation and ROUGE reporting
  inference.py    Reusable summarizer and CLI
  metrics.py      ROUGE metric helpers
  preprocessing.py
  train.py        Seq2SeqTrainer fine-tuning pipeline
tests/
  test_preprocessing.py
```

To run the lightweight, dependency-free tests:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```

## Notes

- Model and dataset downloads require access to Hugging Face on the first run.
- The default Pegasus checkpoint is a general summarization checkpoint; SAMSum fine-tuning adapts it to conversational dialogue.
- Keep generated checkpoints and downloaded datasets out of source control. The scripts do not upload models or data.
