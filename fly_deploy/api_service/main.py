"""
API / Orchestrator service.

Routes:
  POST /predict      — classify text, store result in DB
  GET  /health       — liveness probe
  GET  /ready        — readiness probe (checks classifier + DB)
  GET  /predictions/recent — last N predictions from DB
"""
from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any
from uuid import UUID, uuid4

import asyncpg
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLASSIFIER_URL = os.getenv("CLASSIFIER_URL", "http://localhost:8001")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://app:app@localhost:5432/app")
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "5.0"))
DB_RETRIES = int(os.getenv("DB_CONNECT_RETRIES", "30"))
DB_SLEEP = float(os.getenv("DB_CONNECT_SLEEP_SECONDS", "1.0"))


CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dialog_id       UUID NOT NULL,
    message_id      UUID,
    participant_index INTEGER,
    input_text      TEXT NOT NULL,
    is_bot_probability DOUBLE PRECISION NOT NULL,
    label           TEXT NOT NULL,
    model_name      TEXT,
    model_version   TEXT,
    created_at      TIMESTAMPTZ DEFAULT now()
);
"""


class PredictRequest(BaseModel):
    text: str = Field(min_length=1)
    dialog_id: UUID
    id: UUID | None = None
    participant_index: int = 0


class PredictResponse(BaseModel):
    id: UUID
    dialog_id: UUID
    message_id: UUID | None
    participant_index: int
    is_bot_probability: float


@asynccontextmanager
async def lifespan(app_: FastAPI):
    # Connect to DB with retries (postgres may still be starting)
    pool = None
    for attempt in range(DB_RETRIES):
        try:
            pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
            async with pool.acquire() as conn:
                await conn.execute(CREATE_TABLE)
            logger.info("DB connected and table ready")
            break
        except Exception as exc:
            logger.warning("DB not ready (attempt %d/%d): %s", attempt + 1, DB_RETRIES, exc)
            await asyncio.sleep(DB_SLEEP)
    else:
        logger.error("Could not connect to DB after %d attempts", DB_RETRIES)

    app_.state.db = pool
    yield

    if pool:
        await pool.close()


app = FastAPI(title="Echo Bot API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
async def ready() -> dict[str, Any]:
    issues: list[str] = []

    # Check classifier
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(f"{CLASSIFIER_URL}/ready")
            resp.raise_for_status()
            classifier_status = resp.json()
    except Exception as exc:
        issues.append(f"classifier: {exc}")
        classifier_status = {"status": "unavailable"}

    # Check DB
    db_status = "ok"
    if app.state.db:
        try:
            async with app.state.db.acquire() as conn:
                await conn.fetchval("SELECT 1")
        except Exception as exc:
            issues.append(f"db: {exc}")
            db_status = "unavailable"
    else:
        issues.append("db: pool not initialized")
        db_status = "unavailable"

    status = "ready" if not issues else "degraded"
    return {
        "status": status,
        "classifier": classifier_status,
        "database": db_status,
        "issues": issues,
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest) -> PredictResponse:
    # Call classifier
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.post(
                f"{CLASSIFIER_URL}/score",
                json={"text": body.text},
            )
            resp.raise_for_status()
            clf = resp.json()
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"Classifier unavailable: {exc}")
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"Classifier error: {exc}")

    is_bot_probability: float = clf["is_bot_probability"]
    label: str = clf["label"]
    prediction_id = uuid4()

    # Store in DB
    if app.state.db:
        try:
            async with app.state.db.acquire() as conn:
                await conn.execute(
                    """INSERT INTO predictions
                       (id, dialog_id, message_id, participant_index,
                        input_text, is_bot_probability, label, model_name, model_version)
                       VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)""",
                    prediction_id,
                    body.dialog_id,
                    body.id,
                    body.participant_index,
                    body.text,
                    is_bot_probability,
                    label,
                    clf.get("model_name"),
                    clf.get("model_version"),
                )
        except Exception as exc:
            logger.warning("Failed to store prediction: %s", exc)

    return PredictResponse(
        id=prediction_id,
        dialog_id=body.dialog_id,
        message_id=body.id,
        participant_index=body.participant_index,
        is_bot_probability=is_bot_probability,
    )


@app.get("/predictions/recent")
async def predictions_recent(limit: int = 5) -> list[dict]:
    if not app.state.db:
        return []
    async with app.state.db.acquire() as conn:
        rows = await conn.fetch(
            "SELECT id, dialog_id, input_text, is_bot_probability, label, created_at "
            "FROM predictions ORDER BY created_at DESC LIMIT $1",
            limit,
        )
    return [dict(r) for r in rows]
