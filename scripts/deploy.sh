#!/usr/bin/env bash
# Deploys the gev API to Cloud Run, pointed at the model server from ./scripts/deploy-model.sh.
#
#   ./scripts/deploy.sh
#   MODEL_SERVICE=gev-ar MODEL=google/gemma-4-26B-A4B-it ./scripts/deploy.sh   # must match what that model server serves
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-gev}"
MODEL_SERVICE="${MODEL_SERVICE:-gev-ar-fp8}"
MODEL="${MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic}"
SA="${SERVICE}-runtime@${PROJECT}.iam.gserviceaccount.com"
# An experimental copy of the API can share the main one's keys: SECRET=gev-api-keys SERVICE=gev-scored …
SECRET="${SECRET:-${SERVICE}-api-keys}"
# Comma-separated settings; the default is what runs live. A value can't contain a comma, hence the ";".
EXTRA_ENV="${EXTRA_ENV-GEV_STRATEGY=scored,GEV_PROMPT_ORDER=state-last,GEV_WIDE_CHOICE=1,GEV_TEMPERATURE=choice=4;score=4}"

MODEL_URL="$(gcloud run services describe "$MODEL_SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)' 2>/dev/null || true)"
if [[ -z "$MODEL_URL" ]]; then
  echo "No ${MODEL_SERVICE} service in ${PROJECT}/${REGION}. Run ./scripts/deploy-model.sh first." >&2
  exit 1
fi

echo "Deploying ${SERVICE} to ${PROJECT}/${REGION} (model: ${MODEL} at ${MODEL_URL})"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com --project "$PROJECT"

# A dedicated runtime identity whose only power is invoking the private model server.
if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "${SERVICE}-runtime" --project "$PROJECT" --display-name "gev API runtime"
fi
gcloud run services add-iam-policy-binding "$MODEL_SERVICE" --project "$PROJECT" --region "$REGION" \
  --member "serviceAccount:${SA}" --role roles/run.invoker >/dev/null

# The API is publicly reachable, so callers authenticate with a bearer key kept in Secret Manager.
if ! gcloud secrets describe "$SECRET" --project "$PROJECT" >/dev/null 2>&1; then
  KEY="sk-gev-$(openssl rand -hex 24)"
  printf '%s' "$KEY" | gcloud secrets create "$SECRET" --project "$PROJECT" --data-file=- --replication-policy automatic
  echo "Generated API key (shown once): ${KEY}"
fi
gcloud secrets add-iam-policy-binding "$SECRET" --project "$PROJECT" --member "serviceAccount:${SA}" \
  --role roles/secretmanager.secretAccessor --quiet >/dev/null
# The same marker ./scripts/keys.sh sets when it rolls the APIs onto new keys. --set-env-vars replaces
# every variable, so without this a deploy would silently drop it.
KEYS_VERSION="$(gcloud secrets versions describe latest --secret "$SECRET" --project "$PROJECT" --format 'value(name)')"
KEYS_VERSION="${KEYS_VERSION##*/}"

# --timeout covers a request that arrives while the model server is cold-starting.
gcloud run deploy "$SERVICE" --source . --project "$PROJECT" --region "$REGION" \
  --service-account "$SA" --allow-unauthenticated \
  --cpu 1 --memory 512Mi --concurrency 80 --max-instances 10 --timeout 900 \
  --set-env-vars "GEV_MODEL_URL=${MODEL_URL}/v1,GEV_MODEL=${MODEL},GEV_MODEL_GCP_AUTH=1,GEV_KEYS_VERSION=${KEYS_VERSION}${EXTRA_ENV:+,${EXTRA_ENV}}" \
  --set-secrets "GEV_API_KEYS=${SECRET}:latest"

echo "Deployed: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
