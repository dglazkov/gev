#!/usr/bin/env bash
# Serves a Gemma model with vLLM (or SGLang) on a Cloud Run GPU, as gev's model backend.
#
#   ./scripts/deploy-model.sh                                  # the live server: FP8 Gemma 4 26B-A4B on an RTX PRO 6000
#   SERVICE=gev-ar MODEL=google/gemma-4-26B-A4B-it ./scripts/deploy-model.sh  # any Gemma that vLLM can serve
#   SERVICE=gev-try EXTRA_ARGS=… ./scripts/deploy-model.sh    # an experiment: never on the live service
#   SERVICE=gev-sglang SERVER=sglang ./scripts/deploy-model.sh  # on SGLang; the API then needs GEV_MODEL_SERVER=sglang
#
# Weights are copied from Hugging Face into a Cloud Storage bucket once, then mounted
# read-only into the container, so cold starts never depend on Hugging Face.
set -euo pipefail

# Not the gcloud default: the owner's default project is a different one.
PROJECT="${GOOGLE_CLOUD_PROJECT:-gev-systemone}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-gev-ar-fp8}"
MODEL="${MODEL:-RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic}"
SERVER="${SERVER:-vllm}"
case "$SERVER" in
  vllm) DEFAULT_IMAGE=docker.io/vllm/vllm-openai:v0.29.0 ;;
  # CUDA 13.0, which is what Cloud Run's driver (580) supports. Gemma 4 FP8 needs SGLang v0.5.15 or later.
  sglang) DEFAULT_IMAGE=docker.io/lmsysorg/sglang:v0.5.20 ;;
  *) echo "SERVER must be vllm or sglang, got ${SERVER}" >&2; exit 1 ;;
esac
IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
GPU_TYPE="${GPU_TYPE:-nvidia-rtx-pro-6000}"
CPU="${CPU:-20}"
MEMORY="${MEMORY:-80Gi}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
# A request is one batch of up to a few dozen short prompts, mostly served from the prefix cache.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"

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

# Where vLLM reads the weights: the mounted bucket, or e.g. MODEL_PATH=gs://bucket/path with
# EXTRA_ARGS=--load-format=runai_streamer to stream them instead of reading through the mount.
MODEL_PATH="${MODEL_PATH:-/models/${MODEL}}"
COMMAND=""
if [[ "$SERVER" == sglang ]]; then
  # The image has no entrypoint. SGLang doesn't cap logprobs, so it needs nothing like --max-logprobs,
  # and its memory fraction is left to SGLang: it counts differently from vLLM's utilization.
  # --weight-loader-disable-mmap: read through the mount, SGLang's default mmap loader manages ~10 MB/s
  # and never gets past the 30-minute startup probe; one sequential read makes it ~64 MB/s, like vLLM.
  COMMAND=python3
  ARGS="-m,sglang.launch_server,--model-path=${MODEL_PATH},--served-model-name=${MODEL},--context-length=${MAX_MODEL_LEN}"
  ARGS+=",--max-running-requests=${MAX_NUM_SEQS},--chunked-prefill-size=16384,--weight-loader-disable-mmap,--enable-metrics,--host=0.0.0.0,--port=8000"
else
  ARGS="--model=${MODEL_PATH},--served-model-name=${MODEL},--max-model-len=${MAX_MODEL_LEN},--max-num-seqs=${MAX_NUM_SEQS}"
  # --max-logprobs: a choice answered by name reads the first token of up to ~200 names (GEV_WIDE_CHOICE).
  ARGS+=",--gpu-memory-utilization=${GPU_MEMORY_UTILIZATION},--max-num-batched-tokens=16384,--max-logprobs=256,--host=0.0.0.0,--port=8000"
fi
# Comma-separated extra server flags, e.g. EXTRA_ARGS=--max-num-batched-tokens=16384
[[ -n "${EXTRA_ARGS:-}" ]] && ARGS+=",${EXTRA_ARGS}"
# Comma-separated environment for the container, e.g. EXTRA_ENV=VLLM_LOGGING_LEVEL=DEBUG
ENV_VARS="GEV_DEPLOYED_BY=deploy-model.sh${EXTRA_ENV:+,${EXTRA_ENV}}"
# Cold start is dominated by reading the weights through the Cloud Storage FUSE mount at ~50–60 MB/s
# (26 GiB of FP8: ~7.5 min, plus ~1.7 min of engine init). Measured and slower: vLLM's
# --safetensors-load-strategy=prefetch, and --load-format=runai_streamer from gs://. The container's
# network path to the bucket is the limit, not how the bytes are read.

# Private (--no-allow-unauthenticated): only the gev API's service account may invoke it.
# The startup probe allows 30 minutes for weights to load from the bucket.
# --min-instances 0: a revision never pins a GPU on. Whether the live server is kept warm is its
# service-level minimum, which the Cloud Scheduler jobs gev-warm-on / gev-warm-off switch without a new
# revision (docs/STATE.md); a revision-level minimum would outrank it. Cold, the first request waits
# ~10 minutes for the weights and the ones queued behind it get 429s.
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" \
  --image "$IMAGE" ${COMMAND:+"--command=${COMMAND}"} --args="$ARGS" --set-env-vars "$ENV_VARS" --port 8000 \
  --service-account "$SA" --no-allow-unauthenticated \
  --gpu 1 --gpu-type "$GPU_TYPE" --no-gpu-zonal-redundancy \
  --cpu "$CPU" --memory "$MEMORY" --no-cpu-throttling \
  --min-instances 0 --max-instances 1 --concurrency 64 --timeout 900 \
  --add-volume "name=models,type=cloud-storage,bucket=${BUCKET},readonly=true" \
  --add-volume-mount "volume=models,mount-path=/models" \
  --startup-probe "httpGet.path=/health,httpGet.port=8000,initialDelaySeconds=30,periodSeconds=30,failureThreshold=60,timeoutSeconds=10"

echo "Model server: $(gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
echo "Next: MODEL=${MODEL} ./scripts/deploy.sh"
