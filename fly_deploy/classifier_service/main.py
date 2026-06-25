"""Classifier sidecar — bot/human keyword classifier."""
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    text: str = Field(min_length=1, max_length=5000)


class ScoreResponse(BaseModel):
    is_bot_probability: float = Field(ge=0.0, le=1.0)
    label: str
    model_name: str
    model_version: str


class BotClassifier:
    bot_words = ("bot", "assistant", "automate", "generate", "support", "instant", "help",
                 "click here", "subscribe", "follow", "like", "share", "win", "prize",
                 "limited offer", "act now", "buy now", "free", "discount")
    human_words = ("i think", "i feel", "i believe", "honestly", "personally", "yesterday",
                   "tomorrow", "my friend", "by the way", "anyway", "lol", "haha", "thanks")

    def predict(self, text: str) -> tuple[str, float]:
        norm = text.lower()
        bot_hits = sum(w in norm for w in self.bot_words)
        human_hits = sum(w in norm for w in self.human_words)
        length_bonus = min(len(norm) / 500.0, 0.1)

        bot_prob = min(0.1 + bot_hits * 0.18 + length_bonus - human_hits * 0.08, 0.98)
        bot_prob = max(bot_prob, 0.02)
        label = "bot" if bot_prob >= 0.5 else "human"
        return label, round(bot_prob, 4)


@asynccontextmanager
async def lifespan(app_: FastAPI):
    app_.state.model = BotClassifier()
    app_.state.model_name = os.getenv("MODEL_NAME", "keyword-bot-classifier")
    app_.state.model_version = os.getenv("MODEL_VERSION", "v1")
    yield


app = FastAPI(title="Bot Classifier Sidecar", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict:
    return {
        "status": "ready",
        "model_name": app.state.model_name,
        "model_version": app.state.model_version,
    }


@app.post("/score", response_model=ScoreResponse)
def score(request: ScoreRequest) -> ScoreResponse:
    label, bot_prob = app.state.model.predict(request.text)
    return ScoreResponse(
        is_bot_probability=bot_prob,
        label=label,
        model_name=app.state.model_name,
        model_version=app.state.model_version,
    )
