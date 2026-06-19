# Echo Bot

Echo bot with message classification. The FastAPI server classifies each message using `typeform/distilbert-base-uncased-mnli`, and the Streamlit client shows the bot probability next to each message along with session metrics.

## Setup

```bash
uv sync
```

## Running

Open two terminals:

```bash
# Terminal 1 — FastAPI (port 6872)
uv run fastapi dev app/api/main.py --host 0.0.0.0 --port 6872

# Terminal 2 — Streamlit (port 8502)
PYTHONPATH=$(pwd) uv run streamlit run app/web/streamlit_app.py --server.port 8502
```

Open in browser: http://localhost:8502

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/get_message` | Echo response to a message |
| `POST` | `/predict` | Classify a message, returns `is_bot_probability` |
