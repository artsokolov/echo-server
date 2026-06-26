"""
Orchestrator — public API gateway.

Routes:
  POST /predict     → http://classifier:8000/predict
  POST /get_message → http://llm:11434/v1/chat/completions

The orchestrator does NOT run any ML models itself.
"""
import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import UUID4, BaseModel, Field, StrictStr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://classifier:8000")
LLM_URL = os.getenv("LLM_URL", "http://llm:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")

app = FastAPI(title="Orchestrator Gateway")


class PredictRequest(BaseModel):
    text: StrictStr
    dialog_id: UUID4
    id: UUID4
    participant_index: int = 0

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Of course! I'd be happy to help you with that.",
                    "dialog_id": "550e8400-e29b-41d4-a716-446655440001",
                    "id": "550e8400-e29b-41d4-a716-446655440002",
                    "participant_index": 0,
                },
                {
                    "text": "lol yeah bro idk what u mean tbh",
                    "dialog_id": "550e8400-e29b-41d4-a716-446655440003",
                    "id": "550e8400-e29b-41d4-a716-446655440004",
                    "participant_index": 0,
                },
            ]
        }
    }


class GetMessageRequest(BaseModel):
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: UUID4 | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "dialog_id": "550e8400-e29b-41d4-a716-446655440001",
                    "last_msg_text": "What is the capital of France?",
                    "last_message_id": None,
                }
            ]
        }
    }


class GetMessageResponse(BaseModel):
    new_msg_text: StrictStr
    dialog_id: UUID4


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(body: PredictRequest) -> JSONResponse:
    """Forward /predict requests to the classifier service."""
    logger.info("Routing /predict → %s/predict", CLASSIFIER_URL)
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{CLASSIFIER_URL}/predict", json=body.model_dump(mode="json"))
            resp.raise_for_status()
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text)
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Classifier unavailable: {exc}")


@app.post("/get_message", response_model=GetMessageResponse)
async def get_message(body: GetMessageRequest) -> GetMessageResponse:
    """Forward /get_message to the LLM service (OpenAI-compatible endpoint)."""
    logger.info("Routing /get_message → %s/v1/chat/completions", LLM_URL)
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json={
                    "model": LLM_MODEL,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a friendly assistant. Reply concisely.",
                        },
                        {"role": "user", "content": body.last_msg_text},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 256,
                },
            )
            resp.raise_for_status()
            reply = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("LLM unavailable (%s), returning fallback echo.", exc)
        reply = f"[LLM unavailable] Echo: {body.last_msg_text}"

    return GetMessageResponse(new_msg_text=reply, dialog_id=body.dialog_id)
