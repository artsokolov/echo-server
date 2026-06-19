import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from transformers import pipeline

from app.core.logging import app_logger
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

_classifier = None

# In-memory dialog history: dialog_id -> list of {"text": str, "participant_index": int}
_dialog_history: dict[str, list[dict]] = defaultdict(list)


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


def classify_conversation(messages: list[dict]) -> float:
    classifier = load_model()
    conversation = "\n".join(
        f"{m['participant_index']}: {m['text']}" for m in messages
    )
    prompt = f"Determine if there is an AI bot in the dialog:\n\n{conversation}"
    result = classifier(
        prompt,
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


@app.post("/get_message", response_model=GetMessageResponseModel)
async def get_message(body: GetMessageRequestModel):
    app_logger.info(
        f"Received message dialog_id: {body.dialog_id}, last_msg_id: {body.last_message_id}"
    )
    return GetMessageResponseModel(
        new_msg_text=body.last_msg_text, dialog_id=body.dialog_id
    )


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    dialog_key = str(msg.dialog_id)

    # Store message; skip duplicates (same id) to avoid double-counting echo
    history = _dialog_history[dialog_key]
    if not any(str(msg.id) == str(m.get("id", "")) for m in history):
        history.append(
            {"id": str(msg.id), "text": msg.text, "participant_index": msg.participant_index}
        )

    is_bot_probability = classify_conversation(history)

    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=is_bot_probability,
    )
