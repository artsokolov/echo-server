#!/bin/bash
set -e

REMOTE_HOST="158.160.135.246"
PRIVATE_KEY="portforward_key"
PORT_FILE="/tmp/echobot_port.txt"

if [[ ! -f "$PRIVATE_KEY" ]]; then
  echo "ERROR: $PRIVATE_KEY not found."
  echo "Download it from the quickstart repo:"
  echo "  curl -L https://raw.githubusercontent.com/open-cu/youarebot-quickstart/main/portforward_key -o portforward_key"
  echo "  chmod 600 portforward_key"
  exit 1
fi

# Reuse port across runs so the registration URL stays stable
if [[ -f "$PORT_FILE" ]]; then
  RANDOM_PORT=$(cat "$PORT_FILE")
else
  RANDOM_PORT=$(awk -v min=1024 -v max=65535 'BEGIN{srand(); print int(min+rand()*(max-min+1))}')
  echo "$RANDOM_PORT" > "$PORT_FILE"
fi

chmod 600 "$PRIVATE_KEY"

echo "Opening SSH reverse tunnel on port $RANDOM_PORT..."
ssh -f -i "$PRIVATE_KEY" -N \
  -o StrictHostKeyChecking=no \
  -o ServerAliveInterval=30 \
  -R "0.0.0.0:$RANDOM_PORT:localhost:6872" \
  "forwarduser@$REMOTE_HOST"

echo ""
echo "====================================================="
echo "  Register this URL on youare.bot:"
echo "  http://$REMOTE_HOST:$RANDOM_PORT"
echo "====================================================="
echo ""

docker compose up --build
