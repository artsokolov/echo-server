#!/bin/bash
# Start Ollama server in background, pull model, then keep serving
ollama serve &
SERVER_PID=$!

echo "Waiting for Ollama to start..."
until ollama list > /dev/null 2>&1; do sleep 1; done

echo "Pulling model..."
ollama pull qwen2.5:1.5b

echo "Model ready."
wait $SERVER_PID
