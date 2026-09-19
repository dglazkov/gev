# gev

A [jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)-style "System One" decision API, built on GCP: Gemma on a Cloud Run GPU.

You pass in **state** and a set of **typed questions**. gev returns typed answers with probabilities that your code can threshold, rank, or use to defer to a human. It generates no text.

```bash
curl -s "$GEV_URL/v1/systemone" -H "authorization: Bearer $GEV_API_KEY" -H 'content-type: application/json' -d '{
  "state": "I was charged twice and my demo is in two hours. Fix this NOW.",
  "questions": {
    "team":        { "type": "choice", "instructions": "Which team should handle this?",
                     "criteria": { "billing": "Payment issues", "technical": "Bugs", "sales": null } },
    "urgent":      { "type": "noul",   "instructions": "The message conveys urgency" },
    "frustration": { "type": "score",  "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Frustrated but civil", "Very angry"] }
  }
}'
```

```json
{
  "model": "google/diffusiongemma-26B-A4B-it",
  "answers": {
    "team":        { "type": "choice", "choice": "billing", "probabilities": { "billing": 0.97, "technical": 0.02, "sales": 0.01 }, "confidence": 0.86 },
    "urgent":      { "type": "noul", "noul": 0.99 },
    "frustration": { "type": "score", "score": 1.8, "legend": { "0": "Calm", "1": "Frustrated but civil", "2": "Very angry" },
                     "probabilities": { "0": 0.01, "1": 0.18, "2": 0.81 }, "confidence": 0.52 }
  },
  "usage": { "input_tokens": 612, "output_tokens": 3 }
}
```

(Illustrative numbers.) The request and response shapes follow jev's `POST /v1/systemone`, so a jev client pointed at a gev base URL should work. `model` is accepted and ignored.

## How it works

gev doesn't know how real System One models work inside. It approximates the interface with Gemma's token probabilities, following the approach described by Cloudflare's Workers AI team:

1. Each question becomes one prompt: the state, the question, and a list of answers, each tagged with a **single-token label** (`A`…`P` for choice, `0`…`9` for score, `yes`/`no` for noul).
2. The model is asked for exactly **one output token**, at temperature 0, thinking disabled, with the **top-20 logprobs** for that token (hence labels in chunks of 16).
3. Logprobs for each label are exponentiated, pooled across surface variants (`A`, ` a`, `▁A`), and **renormalized** into a distribution. Off-script tokens are dropped. A label that doesn't appear in the top 20 gets half the smallest visible probability rather than zero.
4. All questions run **in parallel**. State comes first in every prompt, so the shared prefix is eligible for vLLM's prefix caching.

Derived fields: `score` is the expected zero-indexed level; `confidence` is 1 − normalized entropy (0 = uniform, 1 = certain); `noul` is P(yes).

Choice questions with more than 16 options (up to jev's 255) run as a tournament: each chunk of 16 is ranked, then the chunk winners are ranked against each other.

**Caveat:** these are an LLM's token probabilities, not calibrated outcomes. They're useful for ranking and thresholding, but measure them on your own data before trusting `0.9` to mean 90%. Letter labels also carry position bias; `GEV_ROTATIONS=3` asks each choice question with rotated option orders and averages, at 3× the calls.

## Architecture

```
client ──▶ Cloud Run: gev API (this repo, CPU) ──▶ Cloud Run GPU: vLLM serving Gemma (private, scale to zero)
                                                        └── weights mounted read-only from Cloud Storage
```

| Cloudflare demo       | gev                                                              |
| --------------------- | ---------------------------------------------------------------- |
| Worker                | Cloud Run service (Node 24, Hono)                                |
| Workers AI GPU        | Cloud Run GPU (RTX PRO 6000 Blackwell, 96 GB), vLLM              |
| DiffusionGemma        | DiffusionGemma (`google/diffusiongemma-26B-A4B-it`), swappable   |

The model server is private: only the API's service account can invoke it, using a Google ID token. It scales to zero, so idle costs nothing, and the first request after idle waits for a cold start while vLLM loads ~52 GB of weights from the bucket.

gev talks to the model through vLLM's OpenAI-compatible API (`max_tokens: 1`, `top_logprobs: 20`, thinking disabled), so the model is a deploy-time variable. Anything vLLM serves works, e.g. `google/gemma-4-26B-A4B-it` (autoregressive, same backbone) or `google/gemma-4-E4B-it` on an L4.

Today each question is one request. vLLM has an open PR ([#57250](https://github.com/vllm-project/vllm/pull/57250)) adding a jev-like structured mode for DiffusionGemma that answers many questions on one canvas in a single denoising pass; once it lands, that becomes a second `Backend` implementation here.

## Deploy

```bash
export GOOGLE_CLOUD_PROJECT=my-project
./scripts/deploy-model.sh    # bucket + one-time weight copy (Cloud Build) + vLLM on a Cloud Run GPU
./scripts/deploy.sh          # the gev API, wired to the model server; prints an API key once
```

`deploy-model.sh` takes `MODEL`, `GPU_TYPE`, `CPU`, `MEMORY`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `IMAGE`, `REGION`. RTX PRO 6000 GPUs need quota in the region (us-central1 by default) and a minimum of 20 vCPU / 80 GiB. For a small model on an L4:

```bash
MODEL=google/gemma-4-E4B-it GPU_TYPE=nvidia-l4 CPU=8 MEMORY=32Gi ./scripts/deploy-model.sh
MODEL=google/gemma-4-E4B-it ./scripts/deploy.sh
```

## Run locally

Requires Node 24+ (it runs the TypeScript directly; there is no build step). Tunnel to the private model server, then point gev at it:

```bash
npm install
gcloud run services proxy gev-model --region us-central1 --port 8000 &
GEV_MODEL_URL=http://localhost:8000/v1 npm run dev     # http://localhost:8080 — demo page at /
npm test && npm run typecheck
```

Any OpenAI-compatible server that returns `top_logprobs` works as `GEV_MODEL_URL`, including a local vLLM.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `GEV_MODEL_URL` | (required) | Model server base URL, e.g. `https://gev-model-….run.app/v1` |
| `GEV_MODEL` | `google/diffusiongemma-26B-A4B-it` | Must match the model server's served model name |
| `GEV_MODEL_GCP_AUTH` | | `1` to call a private Cloud Run model server with a Google ID token |
| `GEV_MODEL_API_KEY` | | Bearer key, if the model server uses one instead |
| `GEV_API_KEYS` | (open) | Comma-separated bearer keys clients must present |
| `GEV_CONCURRENCY` | `16` | Max in-flight model calls per request |
| `GEV_ROTATIONS` | `1` | Rotated re-asks per choice question, averaged |

## Errors

`401` bad API key · `422` validation failed (with `issues`) · `429` model backend rate-limited · `502` model backend failed. Retry `429`/`502` with exponential backoff.
