#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-${BASE_URL:-}}"

if [[ -z "$BASE_URL" ]]; then
  if [[ -n "${COMPOSE_APP:-}" ]]; then
    BASE_URL="https://$COMPOSE_APP.fly.dev"
  else
    echo "Usage: $0 <base-url>"
    echo "   or: export COMPOSE_APP=<name>"
    exit 1
  fi
fi

echo "Testing $BASE_URL ..."

wait_for_json() {
  local path="$1"
  local max=45
  for i in $(seq 1 $max); do
    if curl -fsS "$BASE_URL$path" -o /dev/null 2>/dev/null; then
      echo "  OK $path"
      return 0
    fi
    echo "  Waiting for $path ($i/$max)..."
    sleep 2
  done
  echo "  TIMEOUT waiting for $path" >&2
  return 1
}

echo ""
echo "=== Waiting for /health ==="
wait_for_json "/health"

echo ""
echo "=== /health ==="
curl -fsS "$BASE_URL/health" | python3 -m json.tool

echo ""
echo "=== /ready ==="
curl -fsS "$BASE_URL/ready" | python3 -m json.tool

echo ""
echo "=== POST /predict ==="
curl -fsS -X POST "$BASE_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, I want to buy a cheap iPhone!",
    "dialog_id": "550e8400-e29b-41d4-a716-446655440001",
    "id": "550e8400-e29b-41d4-a716-446655440002",
    "participant_index": 0
  }' | python3 -m json.tool

echo ""
echo "=== GET /predictions/recent ==="
curl -fsS "$BASE_URL/predictions/recent?limit=3" | python3 -m json.tool

echo ""
echo "All tests passed!"
