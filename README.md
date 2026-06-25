# Echo Bot — Microservice Architecture

Bot/human classifier with LLM echo responses, structured as a microservice system.

## Architecture

```
                        ┌─────────────────────────────┐
                        │         Streamlit UI         │
                        │         localhost:8502        │
                        └──────────────┬──────────────┘
                                       │
                        ┌──────────────▼──────────────┐
                        │    Orchestrator (gateway)    │
                        │         port 6872            │
                        └───────┬──────────┬──────────┘
                                │          │
               POST /predict    │          │  POST /get_message
                                │          │
               ┌────────────────▼──┐   ┌──▼───────────────────┐
               │    Classifier      │   │       LLM (Ollama)    │
               │  port 8001 (ext)  │   │     port 11434        │
               │  port 8000 (int)  │   │  POST /v1/chat/...    │
               └────────┬──────────┘   └──────────────────────┘
                        │
               ┌────────▼──────────┐
               │      MLflow       │
               │     port 5000     │
               │  Tracking server  │
               └───────────────────┘
```

| Service | Description | Internal port | External port |
|---------|-------------|--------------|---------------|
| `mlflow` | Experiment tracking & model registry | 5000 | 5000 |
| `classifier` | FastAPI — bot/human classifier | 8000 | 8001 |
| `llm` | Ollama — OpenAI-compatible LLM | 11434 | 11434 |
| `orchestrator` | FastAPI gateway — routes requests | 8000 | 6872 |
| `streamlit` | Chat UI | 8502 | 8502 |

**Routing rules (orchestrator):**

| Public endpoint | Forwards to |
|-----------------|-------------|
| `POST /predict` | `http://classifier:8000/predict` |
| `POST /get_message` | `http://llm:11434/v1/chat/completions` |

The orchestrator does **not** run any ML models itself.

---

## Quick start

```bash
docker compose up --build
```

First run downloads the Qwen2.5-1.5B model (~1 GB) into the `ollama_data` volume.  
Wait for all services to be healthy before testing (~3–5 min on first run).

Open the chat UI: **http://localhost:8502**

---

## Test the endpoints

### Check health

```bash
curl http://localhost:6872/health
# {"status":"ok"}
```

### POST /predict — bot probability

```bash
curl -s -X POST http://localhost:6872/predict \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how are you?",
    "dialog_id": "00000000-0000-0000-0000-000000000001",
    "id": "00000000-0000-0000-0000-000000000002",
    "participant_index": 0
  }' | python3 -m json.tool
```

Expected: `"is_bot_probability"` in `[0.0, 1.0]`.

### POST /get_message — LLM response

```bash
curl -s -X POST http://localhost:6872/get_message \
  -H "Content-Type: application/json" \
  -d '{
    "dialog_id": "00000000-0000-0000-0000-000000000001",
    "last_msg_text": "What is the capital of France?",
    "last_message_id": null
  }' | python3 -m json.tool
```

Expected: `"new_msg_text"` with a real LLM answer.

### MLflow UI

Open **http://localhost:5000** to browse experiment runs.

---

## Running locally (dev, no Docker)

```bash
uv sync

# Terminal 1 — FastAPI (original monolith, for dev only)
uv run fastapi dev app/api/main.py --host 0.0.0.0 --port 6872

# Terminal 2 — Streamlit
PYTHONPATH=$(pwd) uv run streamlit run app/web/streamlit_app.py --server.port 8502

# Terminal 3 — MLflow UI (uses existing experiment data)
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

---

## ML experiments (Session 11 homework)

The `notebooks/` directory contains LoRA fine-tuning experiments on SST-2:

```bash
# Run all 4 experiments (takes ~2 min on CPU)
uv run python notebooks/run_experiments.py

# View results
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

See `notebooks/mlflow_lora_experiments.ipynb` for the full notebook version.

---

## Public tunnel (youare.bot)

```bash
# One-time: download SSH key
curl -L https://raw.githubusercontent.com/open-cu/youarebot-quickstart/main/portforward_key \
  -o portforward_key && chmod 600 portforward_key

./run.sh   # prints public URL to register on youare.bot
```

The tunnel forwards to port 6872 (the orchestrator).

---

## Secrets

No secrets, tokens, or API keys are committed.  
All configuration is via environment variables defined in `docker-compose.yaml`.
