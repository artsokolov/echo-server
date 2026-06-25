#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

: "${IMAGES_APP:?Set IMAGES_APP, e.g.: export IMAGES_APP=anri-puankar-gmail-com-harbour-images}"

FLY_ORG="${FLY_ORG:-harbour-ml-solution-course}"
CLASSIFIER_IMAGE_TAG="${CLASSIFIER_IMAGE_TAG:-classifier-v1}"
API_IMAGE_TAG="${API_IMAGE_TAG:-api-v1}"
CLASSIFIER_IMAGE="registry.fly.io/$IMAGES_APP:$CLASSIFIER_IMAGE_TAG"
API_IMAGE="registry.fly.io/$IMAGES_APP:$API_IMAGE_TAG"

fly auth whoami >/dev/null

# Create images app if it doesn't exist
if ! fly apps list --org "$FLY_ORG" --quiet 2>/dev/null | awk '{print $1}' | grep -qx "$IMAGES_APP"; then
  fly apps create "$IMAGES_APP" --org "$FLY_ORG" --yes
fi

fly auth docker

echo "Building classifier image for linux/amd64..."
docker build --platform linux/amd64 -t "$CLASSIFIER_IMAGE" ./classifier_service
echo "Pushing $CLASSIFIER_IMAGE ..."
docker push "$CLASSIFIER_IMAGE"

echo "Building api image for linux/amd64..."
docker build --platform linux/amd64 -t "$API_IMAGE" -f Dockerfile .
echo "Pushing $API_IMAGE ..."
docker push "$API_IMAGE"

echo ""
echo "Done! Images pushed."
echo "export CLASSIFIER_IMAGE=$CLASSIFIER_IMAGE"
echo "export API_IMAGE=$API_IMAGE"
