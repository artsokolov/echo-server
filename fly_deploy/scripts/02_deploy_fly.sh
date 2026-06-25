#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${COMPOSE_APP:?Set COMPOSE_APP, e.g.: export COMPOSE_APP=anri-puankar-gmail-com-harbour-compose}"
: "${IMAGES_APP:?Set IMAGES_APP, e.g.: export IMAGES_APP=anri-puankar-gmail-com-harbour-images}"

FLY_REGION="${FLY_REGION:-fra}"
CLASSIFIER_IMAGE_TAG="${CLASSIFIER_IMAGE_TAG:-classifier-v1}"
API_IMAGE_TAG="${API_IMAGE_TAG:-api-v1}"
CLASSIFIER_IMAGE="${CLASSIFIER_IMAGE:-registry.fly.io/$IMAGES_APP:$CLASSIFIER_IMAGE_TAG}"
API_IMAGE="${API_IMAGE:-registry.fly.io/$IMAGES_APP:$API_IMAGE_TAG}"
FLY_CONFIG="fly.generated.toml"
FLY_COMPOSE_FILE="docker-compose.fly.yml"

echo "Generating fly config from templates..."
sed -e "s|__COMPOSE_APP__|$COMPOSE_APP|g" \
    -e "s|__FLY_REGION__|$FLY_REGION|g" \
    fly.toml.template > "$FLY_CONFIG"

sed -e "s|__CLASSIFIER_IMAGE__|$CLASSIFIER_IMAGE|g" \
    -e "s|__API_IMAGE__|$API_IMAGE|g" \
    docker-compose.fly.yml.template > "$FLY_COMPOSE_FILE"

echo "Validating config..."
fly config validate --config "$FLY_CONFIG"

echo "Deploying to Fly.io..."
fly deploy --config "$FLY_CONFIG" --ha=false

echo ""
fly status --app "$COMPOSE_APP"
echo ""
echo "Public URL: https://$COMPOSE_APP.fly.dev"
