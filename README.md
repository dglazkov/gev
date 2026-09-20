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
  "model": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
  "answers": {
    "team":        { "type": "choice", "choice": "billing", "probabilities": { "billing": 0.97, "technical": 0.02, "sales": 0.01 }, "confidence": 0.86 },
    "urgent":      { "type": "noul", "noul": 0.99 },
    "frustration": { "type": "score", "score": 1.8, "legend": { "0": "Calm", "1": "Frustrated but civil", "2": "Very angry" },
                     "probabilities": { "0": 0.01, "1": 0.18, "2": 0.81 }, "confidence": 0.52 }
  },
  "usage": { "input_tokens": 612, "output_tokens": 3 }
}
```

(Illustrative numbers.) Calling a running gev: [docs/API.md](docs/API.md). The request and response shapes follow jev's `POST /v1/systemone`, so a jev client pointed at a gev base URL should work. `model` is accepted and ignored.

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
| DiffusionGemma        | Gemma 4 26B-A4B, autoregressive, FP8 (`RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`), swappable |

The model server is private: only the API's service account can invoke it, using a Google ID token. It keeps one instance warm, because a cold start reads ~26 GB of weights from the bucket and takes about ten minutes.

gev talks to the model through vLLM's OpenAI-compatible API, so the model is a deploy-time variable. Anything autoregressive that vLLM serves works with the `scored` strategy, e.g. `google/gemma-4-26B-A4B-it` (the same model unquantized) or `google/gemma-4-E4B-it` (faster, less accurate).

There are three strategies, chosen with `GEV_STRATEGY`. **scored** (what runs live) sends every question of a request to the model as one batch of single-token completions and reads each answer from the first token's logprobs: one forward pass, nothing generated, answers independent of each other. The state goes last in each prompt (`GEV_PROMPT_ORDER=state-last`) so the question text, which an app repeats on every request, is served from vLLM's prefix cache. A choice with more options than letters is answered by option name (`GEV_WIDE_CHOICE=1`). **isolated** makes one ordinary model call per question. **packed** puts the questions on one answer sheet (`Q1: D`, `Q2: no`, …); it was built for a diffusion model, which fills the sheet in parallel.

**Status: experimental, at jev's latency** (18 questions in ~130 ms from a laptop, 28 questions including a 186-option choice in ~170 ms; jev: ~150 and ~180). Measurements, known vLLM and Cloud Run problems, dead ends, and open decisions are in [docs/STATE.md](docs/STATE.md); house rules for contributors (human or AI) are in [CLAUDE.md](CLAUDE.md). `bench/` replays recorded jev traffic against gev and compares answers and timings.

## Deploy

```bash
export GOOGLE_CLOUD_PROJECT=my-project
./scripts/deploy-model.sh    # bucket + one-time weight copy (Cloud Build) + vLLM on a Cloud Run GPU
./scripts/deploy.sh          # the gev API, wired to the model server; prints an API key once
./scripts/keys.sh add alice  # another key, for someone else; also `revoke <name>` and `list`
```

`deploy-model.sh` takes `SERVICE`, `MODEL`, `IMAGE`, `GPU_TYPE`, `CPU`, `MEMORY`, `MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MIN_INSTANCES`, `EXTRA_ARGS`, `EXTRA_ENV`, `MODEL_PATH`, `REGION`; its defaults are the live model server. RTX PRO 6000 GPUs need quota in the region (us-central1 by default) and a minimum of 20 vCPU / 80 GiB. To try something, deploy it as a separate service and point a separate API at it, never the live ones:

```bash
SERVICE=gev-try MIN_INSTANCES=0 MODEL=google/gemma-4-E4B-it ./scripts/deploy-model.sh
SERVICE=gev-scored SECRET=gev-api-keys MODEL_SERVICE=gev-try MODEL=google/gemma-4-E4B-it ./scripts/deploy.sh
```

## Run locally

Requires Node 24+ (it runs the TypeScript directly; there is no build step). Tunnel to the private model server, then point gev at it:

```bash
npm install
gcloud run services proxy gev-ar-fp8 --region us-central1 --port 8000 &
GEV_MODEL_URL=http://localhost:8000/v1 GEV_STRATEGY=scored GEV_PROMPT_ORDER=state-last GEV_WIDE_CHOICE=1 npm run dev     # http://localhost:8080 — demo page at /
npm test && npm run typecheck
```

Any vLLM works as `GEV_MODEL_URL`, including a local one. `scored` uses vLLM's `/tokenize` and `/detokenize` to learn the model's chat template, and `GEV_WIDE_CHOICE` needs the server started with `--max-logprobs=256`.

## Configuration

| Variable | Default | |
| --- | --- | --- |
| `GEV_MODEL_URL` | (required) | Model server base URL, e.g. `https://gev-ar-fp8-….run.app/v1` |
| `GEV_MODEL` | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` | Must match the model server's served model name |
| `GEV_MODEL_GCP_AUTH` | | `1` to call a private Cloud Run model server with a Google ID token |
| `GEV_MODEL_API_KEY` | | Bearer key, if the model server uses one instead |
| `GEV_API_KEYS` | (open) | Comma-separated bearer keys clients must present |
| `GEV_STRATEGY` | `isolated` | `scored` (live: one batched forward pass per request), `isolated` (one model call per question) or `packed` (one answer sheet per request) |
| `GEV_PROMPT_ORDER` | `state-first` | `state-first`, `question-first` or `state-last` (live): the later the state, the more of each prompt comes from the prefix cache |
| `GEV_WIDE_CHOICE` | | `1` (live) to answer a choice with more than 16 options by name in one round instead of as a tournament |
| `GEV_TEMPERATURE` | `1` | Calibration: softens probabilities without changing decisions. One number, or per type; live: `choice=4;score=4` |
| `GEV_CONCURRENCY` | `16` | Max in-flight model calls per request (`isolated` and `packed`) |
| `GEV_ROTATIONS` | `1` | Rotated re-asks per choice question, averaged |

## Errors

`401` bad API key · `422` validation failed (with `issues`) · `429` model backend rate-limited · `502` model backend failed. Retry `429`/`502` with exponential backoff.
