import logging
import os
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

import httpx
from fastapi import FastAPI
from transformers import pipeline

from app.core.logging import app_logger

LLM_URL = os.getenv("LLM_URL", "http://localhost:8080")
from app.models import (
    GetMessageRequestModel,
    GetMessageResponseModel,
    IncomingMessage,
    Prediction,
)

logger = logging.getLogger(__name__)

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."
MAX_CHARS = 512

_classifier = None

# (dialog_id, participant_index) -> list of message texts, in order received
_speaker_history: dict[tuple[str, int], list[str]] = defaultdict(list)
# track seen message ids to avoid duplicates
_seen_ids: set[str] = set()


def load_model():
    global _classifier
    if _classifier is None:
        logger.info("Loading zero-shot-classification pipeline: %s", MODEL_NAME)
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,
        )
        logger.info("Model loaded.")
    return _classifier


def classify_speaker(texts: list[str]) -> float:
    """Classify a single participant based on all their messages so far."""
    classifier = load_model()
    combined = " ".join(texts)[:MAX_CHARS]
    result = classifier(
        combined,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    bot_index = result["labels"].index(CANDIDATE_LABELS[0])
    return float(max(0.0, min(1.0, result["scores"][bot_index])))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    app_logger.info(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )
    reply = await _call_llm(body.last_msg_text)
    return GetMessageResponseModel(new_msg_text=reply, dialog_id=body.dialog_id)


async def _call_llm(message: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": "qwen2.5:1.5b",
                    "messages": [
                        {"role": "system", "content": "You are a friendly assistant. Reply concisely."},
                        {"role": "user", "content": message},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        app_logger.warning(f"LLM unavailable, falling back to echo: {exc}")
        return message


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    msg_id = str(msg.id)
    if msg_id not in _seen_ids:
        _seen_ids.add(msg_id)
        key = (str(msg.dialog_id), msg.participant_index)
        _speaker_history[key].append(msg.text)

    texts = _speaker_history[(str(msg.dialog_id), msg.participant_index)]
    is_bot_probability = classify_speaker(texts)

    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability,
    )
