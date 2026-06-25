# Echo Bot

Echo bot with message classification. The FastAPI server classifies each message using `typeform/distilbert-base-uncased-mnli`, and the Streamlit client shows the bot probability next to each message along with session metrics.

## Setup

```bash
uv sync
```

## Running locally (dev)

Open two terminals:

```bash
# Terminal 1 — FastAPI (port 6872)
uv run fastapi dev app/api/main.py --host 0.0.0.0 --port 6872

# Terminal 2 — Streamlit (port 8502)
PYTHONPATH=$(pwd) uv run streamlit run app/web/streamlit_app.py --server.port 8502
```

Open in browser: http://localhost:8502

## Running with Docker Compose

Starts three services: llama.cpp LLM, FastAPI bot, Streamlit UI.
On first run downloads the Qwen2.5-1.5B model (~1 GB) — wait a few minutes before opening the chat.

```bash
docker compose up --build
```

Open in browser: http://localhost:8502

The SSH tunnel (`run.sh`) still works for exposing the service to youare.bot — it points to port 6872 which is the same FastAPI port.

## Running with public tunnel (for youare.bot)

The platform needs to reach your `/get_message` and `/predict` endpoints over the internet.
`run.sh` opens an SSH reverse tunnel and prints the public URL to register.

```bash
# 1. Download the SSH key from the quickstart repo (one-time)
curl -L https://raw.githubusercontent.com/open-cu/youarebot-quickstart/main/portforward_key \
  -o portforward_key
chmod 600 portforward_key

# 2. Start everything
./run.sh
```

The script will print:

```
Register this URL on youare.bot:
http://158.160.135.246:<port>
```

Register that URL on the platform, then start a chat. After the chat ends,
the platform computes ML metrics from the `is_bot_probability` values
your `/predict` endpoint returned.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/get_message` | Echo response to a message |
| `POST` | `/predict` | Classify a message, returns `is_bot_probability` |
