"""
Classifier microservice.

Loads typeform/distilbert-base-uncased-mnli for zero-shot bot/human classification.
Exposes POST /predict — returns is_bot_probability in [0, 1].
"""
import logging
from collections import defaultdict
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from pydantic import UUID4, BaseModel, StrictStr
from transformers import pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
CANDIDATE_LABELS = ["bot", "human"]
HYPOTHESIS_TEMPLATE = "This message was written by a {}."
MAX_CHARS = 512

_classifier = None
_speaker_history: dict[tuple[str, int], list[str]] = defaultdict(list)
_seen_ids: set[str] = set()


class IncomingMessage(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int


class Prediction(BaseModel):
    id: UUID4
    message_id: UUID4
    dialog_id: UUID4
    participant_index: int
    is_bot_probability: float


def load_model():
    global _classifier
    if _classifier is None:
        logger.info("Loading classifier model: %s", MODEL_NAME)
        _classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=-1,
        )
        logger.info("Classifier ready.")
    return _classifier


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(title="Classifier Service", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(msg: IncomingMessage) -> Prediction:
    msg_id = str(msg.id)
    if msg_id not in _seen_ids:
        _seen_ids.add(msg_id)
        key = (str(msg.dialog_id), msg.participant_index)
        _speaker_history[key].append(msg.text)

    texts = _speaker_history[(str(msg.dialog_id), msg.participant_index)]
    classifier = load_model()
    combined = " ".join(texts)[:MAX_CHARS]
    result = classifier(
        combined,
        candidate_labels=CANDIDATE_LABELS,
        hypothesis_template=HYPOTHESIS_TEMPLATE,
    )
    bot_index = result["labels"].index(CANDIDATE_LABELS[0])
    probability = float(max(0.0, min(1.0, result["scores"][bot_index])))

    logger.info("Predicted is_bot_probability=%.4f for dialog=%s", probability, msg.dialog_id)
    return Prediction(
        id=uuid4(),
        message_id=msg.id,
        dialog_id=msg.dialog_id,
        participant_index=msg.participant_index,
        is_bot_probability=probability,
    )
