#!/usr/bin/env bash
# Serves a Gemma model with vLLM on a Cloud Run GPU (scale to zero), as gev's model backend.
#
#   ./scripts/deploy-model.sh                                  # DiffusionGemma on an RTX PRO 6000
#   MODEL=google/gemma-4-26B-A4B-it ./scripts/deploy-model.sh  # any Gemma that vLLM can serve
#   MODEL=google/gemma-4-E4B-it GPU_TYPE=nvidia-l4 CPU=8 MEMORY=32Gi ./scripts/deploy-model.sh
#
# Weights are copied from Hugging Face into a Cloud Storage bucket once, then mounted
# read-only into the container, so cold starts never depend on Hugging Face.
set -euo pipefail

PROJECT="${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-gev-model}"
MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
# Known issues with vLLM images for DiffusionGemma (2026-09-19):
# - vllm-openai:gemma (June build) predates vLLM #57414: concurrent requests with logprobs fail with
#   HTTP 500 or return another request's logprobs. Keep model calls serial (GEV_CONCURRENCY=1) on it.
# - nightly-a8d1aa9c… has that fix but crashes during warmup in the FlashInfer attention backend
#   ("Boolean value of Tensor with more than one value is ambiguous"), after a full weight load.
IMAGE="${IMAGE:-docker.io/vllm/vllm-openai:gemma}"
GPU_TYPE="${GPU_TYPE:-nvidia-rtx-pro-6000}"
CPU="${CPU:-20}"
MEMORY="${MEMORY:-80Gi}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
# Diffusion state buffers are allocated outside vLLM's budget; higher values OOM (per the vLLM recipe).
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

BUCKET="${BUCKET:-${PROJECT}-models}"
SA="${SERVICE}@${PROJECT}.iam.gserviceaccount.com"

echo "Deploying ${MODEL} as ${SERVICE} to ${PROJECT}/${REGION} on ${GPU_TYPE}"

gcloud services enable run.googleapis.com cloudbuild.googleapis.com storage.googleapis.com --project "$PROJECT"

if ! gcloud storage buckets describe "gs://${BUCKET}" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud storage buckets create "gs://${BUCKET}" --project "$PROJECT" --location "$REGION" --uniform-bucket-level-access
fi

# One-time copy of the weights, Hugging Face → bucket, on a Cloud Build worker (not this machine).
if ! gcloud storage ls "gs://${BUCKET}/${MODEL}/config.json" >/dev/null 2>&1; then
  BUILD_SA="$(gcloud builds get-default-service-account --project "$PROJECT" --format 'value(serviceAccountEmail)')"
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT" \
    --member "serviceAccount:${BUILD_SA##*/}" --role roles/storage.objectAdmin >/dev/null
  CONFIG="$(mktemp)"
  cat >"$CONFIG" <<'EOF'
steps:
  - name: python:3.12-slim
    entrypoint: bash
    args: ["-c", "pip install -q 'huggingface_hub[hf_transfer]' && HF_HUB_ENABLE_HF_TRANSFER=1 hf download $_MODEL --local-dir /workspace/model && rm -rf /workspace/model/.cache"]
  - name: gcr.io/google.com/cloudsdktool/google-cloud-cli:slim
    entrypoint: gcloud
    args: ["storage", "cp", "--recursive", "/workspace/model/*", "gs://$_BUCKET/$_MODEL/"]
options:
  machineType: E2_HIGHCPU_32
  diskSizeGb: 300
  logging: CLOUD_LOGGING_ONLY
timeout: 7200s
EOF
  gcloud builds submit --no-source --project "$PROJECT" --config "$CONFIG" \
    --substitutions "_MODEL=${MODEL},_BUCKET=${BUCKET}"
  rm -f "$CONFIG"
fi

# The runtime identity can read the weights and nothing else.
if ! gcloud iam service-accounts describe "$SA" --project "$PROJECT" >/dev/null 2>&1; then
  gcloud iam service-accounts create "$SERVICE" --project "$PROJECT" --display-name "gev model server runtime"
fi
# A just-created service account takes a few seconds to become visible to IAM bindings.
for attempt in 1 2 3 4 5 6; do
  gcloud storage buckets add-iam-policy-binding "gs://${BUCKET}" --project "$PROJECT" \
    --member "serviceAccount:${SA}" --role roles/storage.objectViewer >/dev/null 2>&1 && break
  [[ "$attempt" == 6 ]] && { echo "Could not grant ${SA} read access to gs://${BUCKET}" >&2; exit 1; }
  sleep $((attempt * 5))
done

ARGS="--model=/models/${MODEL},--served-model-name=${MODEL},--max-model-len=${MAX_MODEL_LEN},--max-num-seqs=${MAX_NUM_SEQS}"
ARGS+=",--gpu-memory-utilization=${GPU_MEMORY_UTILIZATION},--max-logprobs=20,--host=0.0.0.0,--port=8000"
# Comma-separated extra vLLM flags, e.g. EXTRA_ARGS=--max-num-batched-tokens=16384
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=",${EXTRA_ARGS}"
# Comma-separated environment for the container, e.g. EXTRA_ENV=VLLM_LOGGING_LEVEL=DEBUG
ENV_VARS="GEV_DEPLOYED_BY=deploy-model.sh${EXTRA_ENV:+,${EXTRA_ENV}}"
# Cold start is dominated by reading 48 GiB through the Cloud Storage FUSE mount at ~50 MB/s (~14 min).
# vLLM's --safetensors-load-strategy=prefetch was measured and is slower (18 min): the mount's total
# throughput is the limit, not read parallelism.

# Private (--no-allow-unauthenticated): only the gev API's service account may invoke it.
# The startup probe allows 30 minutes for weights to load from the bucket.
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" --args="$ARGS" --set-env-vars "$ENV_VARS" --port 8000 \
  --service-account "$SA" --no-allow-unauthenticated \
  --gpu 1 --gpu-type "$GPU_TYPE" --no-gpu-zonal-redundancy \
  --cpu "$CPU" --memory "$MEMORY" --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 64 --timeout 900 \
  --add-volume "name=models,type=cloud-storage,bucket=${BUCKET},readonly=true" \
  --add-volume-mount "volume=models,mount-path=/models" \
  --startup-probe "httpGet.path=/health,httpGet.port=8000,initialDelaySeconds=30,periodSeconds=30,failureThreshold=60,timeoutSeconds=10"

echo "Model server: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
echo "Next: MODEL=${MODEL} ./scripts/deploy.sh"
