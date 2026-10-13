"""FastAPI endpoint for dialogue summarization."""

import os
from functools import lru_cache
from typing import Annotated

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import TrainingConfig
from .inference import Summarizer

app = FastAPI(
    title="Dialogue Summarizer API",
    description="Generate a concise summary from a dialogue.",
    version="0.1.0",
)


# Validate the JSON request body and cap its size before model inference.
class SummarizeRequest(BaseModel):
    dialogue: Annotated[str, Field(min_length=1, max_length=50_000)]


# Keep the API response shape stable for clients.
class SummarizeResponse(BaseModel):
    summary: str


@lru_cache(maxsize=1)
def _get_summarizer() -> Summarizer:
    # Allow users to point the API at a different saved model via environment variable.
    model_path = os.environ.get("SUMMARIZER_MODEL", TrainingConfig.output_dir)
    # Cache the model so it is loaded once instead of once per request.
    return Summarizer(model_path)


@app.get("/health")
def health() -> dict[str, str]:
    # Report that the web server is reachable without loading the large model.
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizeResponse)
def summarize(request: SummarizeRequest) -> SummarizeResponse:
    try:
        # Run generation and wrap the plain text in the documented JSON response.
        summary = _get_summarizer().summarize(request.dialogue)
    except ValueError as error:
        # Convert invalid input into the HTTP validation response expected by clients.
        raise HTTPException(status_code=422, detail=str(error)) from error
    return SummarizeResponse(summary=summary)
