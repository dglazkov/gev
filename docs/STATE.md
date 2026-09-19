# State of gev

Last updated 2026-09-19. Keep this current: update it in the same commit as any change that makes
part of it untrue. House rules are in [CLAUDE.md](../CLAUDE.md).

## 1. Objective

Match [jev](https://docs.typesafe.ai/)'s latency with an open model on GCP.

| | jev | gev live (DiffusionGemma, packed) | **experiment: autoregressive Gemma 4, `scored`** | target |
|---|---|---|---|---|
| 18 questions (jtbd), from a laptop, median / p90 | 151 / 255 ms | ~376 ms | **26B-A4B: 149 / 237 ms · E4B: 122 / 135 ms** | ~150 ms |
| 28 questions incl. 186 icons (plan), median / p90 | 181 / 273 ms | ~3,300 ms | **26B-A4B: 265 / 331 ms · E4B: 189 / 266 ms** | ~180 ms |
| model server, per request | ~110 ms | ~220–310 ms | 26B-A4B: 58 ms · E4B: 34 ms (jtbd) | ~110 ms |

**2026-09-19: the latency objective is met on an experimental service** (`gev-scored`, section 2) by
dropping diffusion for a single batched forward pass of an autoregressive model (section 3, `scored`;
section 4, "Scored"). Not yet live; the owner decides (section 7).

The owner: "The key objective of this experiment is to get to the same latency as jev. If we can't,
the point of this project is moot." Accuracy is already on par (section 4); calibration is not.

## 2. What is deployed

GCP project `gev-systemone` (org glazkov.com, Personal Billing), region `us-central1`.

| Service | What | Notes |
|---|---|---|
| `gev` | The API, `https://gev-huio5ftumq-uc.a.run.app` | Public, bearer key (`GEV_API_KEY` in `.env`; Secret Manager `gev-api-keys`), CORS open on `/v1/*`. Demo page at `/`. |
| `gev-model` | vLLM + DiffusionGemma, private | `https://gev-model-huio5ftumq-uc.a.run.app`. RTX PRO 6000 (96 GB), 20 vCPU / 80 GiB, scale to zero, max 1 instance. Serving revision `gev-model-00004-4fx`: image `vllm-openai:gemma`. |
| `gev-ar`, `gev-ar-e4b` | **Experiment (2026-09-19):** autoregressive Gemma 4 on stock vLLM `v0.29.0`, private | `google/gemma-4-26B-A4B-it` and `google/gemma-4-E4B-it`, each on its own RTX PRO 6000, scale to zero, `--max-num-seqs 128 --max-num-batched-tokens 16384`. Stock v0.29.0 starts fine on this GPU with these models. Delete whichever is not chosen. GPU quota in us-central1 is 3. |
| `gev-scored` | **Experiment:** a copy of the API with `GEV_STRATEGY=scored`, `GEV_PROMPT_ORDER=state-last`, `https://gev-scored-huio5ftumq-uc.a.run.app` | Same bearer keys as `gev` (shares the `gev-api-keys` secret). Currently points at `gev-ar` (26B-A4B). Repoint: `SERVICE=gev-scored SECRET=gev-api-keys MODEL_SERVICE=… MODEL=… EXTRA_ENV=GEV_STRATEGY=scored,GEV_PROMPT_ORDER=state-last ./scripts/deploy.sh`. |
| `gs://gev-systemone-models` | Weights | `google/diffusiongemma-26B-A4B-it` (ungated, Apache 2.0), mounted read-only at `/models`. |

**The `gev` service is running a stale build.** It was deployed before the packed strategy and before
the fix for DiffusionGemma's unclosed thought channel (section 5), so its isolated answers may
occasionally come back uniform. It runs with `GEV_CONCURRENCY=1` as a stopgap for the vLLM concurrency
bug. Redeploying it (`scripts/deploy.sh`, ~2 minutes, no model reload) is safe and is waiting on the
owner's go-ahead. `GEV_STRATEGY` defaults to `isolated`; nothing in production uses `packed` yet.

Service accounts: `gev-runtime@` (may only invoke `gev-model` and read its key secret), `gev-model@`
(may only read the bucket).

## 3. How it works

One API request carries a state and N typed questions (`choice`, `score`, `noul`). Three strategies
(`src/engine.ts`, chosen by `GEV_STRATEGY`):

- **isolated**: one model call per question. The model emits one label token (`A`…`P`, `0`…`9`,
  `yes`/`no`); gev normalizes the top-20 logprobs over the labels. Choices with more than 16 options
  run as a tournament of 16-option rounds (jev2ui's 186-icon question = 13 calls).
- **packed**: every question that fits goes on one answer sheet; the model writes `Q1: D`, `Q2: no`, …
  and gev reads the logprobs at each answer position (`src/sheet.ts`). A diffusion model refines all
  lines in parallel, so a sheet costs about one question. Oversized choices stay isolated; any line
  the model skips is re-asked alone ("repair"). Responses carry a non-jev
  `gev: { strategy, model_calls, repaired }` field.

- **scored** (autoregressive models only): the isolated prompts, but all of a request's prompts go to
  vLLM's raw completions endpoint as one batch with `max_tokens: 1`. One forward pass, no generation;
  answers stay independent. A tournament adds one batch per extra round. `GEV_PROMPT_ORDER`
  (`state-first` | `question-first` | `state-last`) moves the STATE later in the prompt so the
  question text, which is the same on every request, is served from the prefix cache. The workload
  makes this matter: in both suites the state is ~15 tokens and the questions ~1,900.
  `bench/model-probe.ts score` measures accuracy and timing for each order.

Derived fields: `score` = expected zero-indexed level; `confidence` = 1 − normalized entropy (jev's
formula is undocumented and differs: for [0, 0.03, 0.97] jev says 0.95, gev 0.88); `noul` = P(yes).

## 4. Measurements

All on DiffusionGemma 26B-A4B, vLLM `gemma` image, RTX PRO 6000, client = a laptop ~58 ms RTT from
us-central1 (jev's API is ~41 ms RTT from the same laptop). Fixtures are real jev2ui traffic recorded
through `bench/record.ts`, with jev's own answers and timings.

**Quality on jev2ui's hand-labeled probe** (40 prompts × 18 questions, labels in `bench/labels/jtbd.json`):

| | stage | done | highStakes | median ms | confidence right / wrong | top choice > 0.99 |
|---|---|---|---|---|---|---|
| jev | 36/40 | 29/40 | 38/40 | 151 | 0.87 / 0.64 | 16% |
| gev isolated (serial) | 36/40 | 32/40 | 37/40 | 3,400 | 0.95 / 0.85 | 57% |
| gev packed | 35/40 | 30/40 | 38/40 | 376 | 1.00 / 0.96 | 92% |

Packed: 0 repairs in 1,084 answers across both suites. On jev2ui's screen plan (13 prompts × 28
questions) packed and isolated make the same decision on 323/364 answers; archetype agrees with jev
10/13 (packed) and 9/13 (isolated). Full plan request: 3.3 s packed, of which ~2.8 s is the 186-icon
tournament (13 serial calls); the other 27 questions take ~445 ms.

**Scored (autoregressive Gemma 4, one batched forward pass), 2026-09-19.** Hand labels and model-server
time from `bench/model-probe.ts score jtbd` (laptop → model server); client times and agreement with
jev from `bench/compare.ts` through `gev-scored` (laptop → API → model server, the real path).

| jtbd, 40 × 18 | stage | done | highStakes | server ms | cache hit | confidence right / wrong | top choice > 0.99 (labeled) |
|---|---|---|---|---|---|---|---|
| jev | 36 | 29 | 38 | ~110 | | 0.87 / 0.64 | 30% |
| 26B-A4B state-first | 36 | 26 | 37 | 85 | 59% | 0.99 / 0.97 | 93% |
| 26B-A4B question-first | 34 | 30 | 37 | 65 | 81% | 0.99 / 0.94 | 90% |
| **26B-A4B state-last** | **38** | **29** | **39** | **58** | 87% | 0.99 / 0.99 | 95% |
| E4B state-first | 32 | 25 | 35 | 57 | 60% | 0.95 / 0.78 | 56% |
| E4B question-first | 33 | 26 | 38 | 41 | 82% | 0.95 / 0.80 | 59% |
| E4B state-last | 33 | 26 | 39 | 34 | 88% | 0.96 / 0.80 | 63% |

| through `gev-scored`, state-last | jtbd median / p90 | plan median / p90 | agrees with jev: jtbd | plan |
|---|---|---|---|---|
| jev | 151 / 255 | 181 / 273 | | |
| 26B-A4B | 149 / 237 | 265 / 331 | 78% (560/720) | 85% (309/364), archetype 8/13 |
| E4B | 122 / 135 | 189 / 266 | 73% (527/720) | 85% (310/364), archetype 11/13 |
| DiffusionGemma packed, for reference | ~376 | ~3,300 | 79% (568/720) | |

- Moving the STATE after the question costs nothing measurable in accuracy with isolated prompts (unlike
  the packed sheet, section 5) and is the fastest, because only ~25 tokens per question are computed.
  Differences of 1–3 labels out of 40 between orders are within noise.
- The plan request is two batches (tournament round one with everything else, then the final round):
  40 prompts, ~12,000 prompt tokens, 90% cached.
- The 26B-A4B is on par with jev on hand labels but **badly overconfident** (worse than packed). E4B is
  less accurate and better spread. Neither is calibrated like jev.
- Isolated prompts repeat the system text per question: 3,900 prompt tokens per jtbd request against
  1,940 packed. Cached, so it costs little.

**Where a packed 18-question request's time goes** (`bench/model-probe.ts latency`, vLLM's own timers):

| | ms |
|---|---|
| server total, new state each time (production-like) | ~300 |
| server total, prompt fully cached | ~216 |
| → prefill of the 1,940-token prompt: uncached vs cached | ~90 vs ~10 |
| → denoising: ~4.5–5 steps × ~45 ms | ~200 |
| queueing | 0 |
| client − server (laptop network, ~100 KB logprobs payload) | ~165 |

**Conclusion at the time: parity needs ~1–2 denoising steps** (superseded: at one step diffusion does
nothing an autoregressive forward pass doesn't, and the forward pass needs no canvas; see "Scored" above).
Original reasoning: parity needs ~1–2 denoising steps (1 step ≈ 60–120 ms server, 2 ≈ 105–165 ms). Prefill
is second. Everything else is noise.

## 5. Learnings

**DiffusionGemma**
- Even with thinking disabled it opens every reply with an empty thought channel,
  `<|channel>thought\n<channel|>`, and **about a third of the time omits the closing marker**.
  `answerPositions` in `src/backends/vllm.ts` strips the preamble token by token.
- Temperature 0 is not deterministic; probabilities wobble in the third decimal between runs.
- It follows the sheet format perfectly at the default step count (0 repairs in 1,084 answers).
- Probabilities are overconfident, and more so when packed: each slot is read after the rest of the
  sheet has settled. open-jev reports the same ("uncalibrated"). jev2ui thresholds on probabilities
  (`JEV_FLOOR = 0.35`, `noul >= 0.6`), so this matters to it.
- **Prompt order trades accuracy for speed.** Questions-before-state raises the prefix-cache hit rate
  from 5% to 89% and saves ~60 ms, but label accuracy drops (stage 35→30, done 30→24 of 40). Kept
  state-first.

**Autoregressive scoring**
- The workload is a ~15-token state against ~1,900 tokens of questions that are constant per app. That
  is what makes prefix caching with the state last so effective.
- **Gemma 4 chat templates differ between models**: E4B's has no empty thought channel in the generation
  prompt, the 26B-A4B's (July revision) has; both put a space after the system text; vLLM's completions
  endpoint adds no `<bos>`. A hand-written frame made E4B answer "The". `VllmBackend` asks the server
  for its rendering once (`/tokenize` with messages, then `/detokenize`) and the prompts are
  token-identical to chat completions.
- A new IAM grant takes a minute or two to work: the first calls from a freshly deployed API to a
  freshly granted model service fail with 401/403.

**vLLM**
- `vllm/vllm-openai:gemma` (June 10 build, v0.22.1rc1) works but predates vLLM #57414: **concurrent
  requests with logprobs fail with HTTP 500 "list index out of range" (~25–40% at 4–12 concurrent) or
  return another request's logprobs.** Reproduce with `bench/model-probe.ts concurrency`.
- `nightly-a8d1aa9c…` (v0.29.1rc1) contains that fix but **crashes during warmup** in the FlashInfer
  attention backend (`RuntimeError: Boolean value of Tensor with more than one value is ambiguous`,
  `flashinfer.py` `build`), after a full weight load. The `gemma` image uses `TRITON_ATTN`. Untried:
  forcing `--attention-backend TRITON_ATTN` on a nightly.
- Prefix caching works out of the box. All diffusion work is reported as "prefill"; read
  `vllm:diffusion_num_denoising_steps_total` for the step count.
- **No per-request control of denoising steps.** `vllm_xargs` `max_denoising_steps`,
  `diffusion_max_steps`, `diffusion_entropy_bound` and a top-level `max_denoising_steps` are all
  accepted and ignored (`bench/model-probe.ts overrides`). The model's `generation_config.json` has
  `max_denoising_steps = 48`, `sampler_config.entropy_bound = 0.1`. The documented server-level knob
  is `--hf-overrides '{"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":X}'` (untried).
- vLLM PR #57250 (unmerged) adds what we actually want: a seeded canvas, a step cap, exact-token
  logprobs at every canvas position.
- SGLang's DiffusionGemma path rejects logprobs outright.

**Cloud Run GPU**
- RTX PRO 6000 quota was available on a brand-new project in us-central1 (the only US region with it).
- **Cold start ≈ 18 minutes**, ~14 of them reading 48 GiB through the Cloud Storage FUSE mount at
  ~50 MB/s regardless of read parallelism. Cloud Build moved the same bytes HF → bucket in 4 minutes.
- Idle instances are reclaimed after ~10 minutes. With max 1 instance this makes timing runs fragile.
- A failed newest revision is retried indefinitely (a full GPU load each time) and cannot be deleted
  until a newer revision exists.
- `gcloud run deploy --source` uploads everything not in `.gcloudignore`, including `.env` if absent.

**jev and jev2ui**
- jev's wire format matches its docs; gev's types mirror it. jev rounds probabilities to 2 places.
- The TypeSafe SDK honors `TYPESAFE_BASE_URL`, sends `Authorization: Bearer`, times out at 10 s with
  2 retries. jev2ui needs no code change to talk to gev.
- jev2ui sends several requests per screen: a 28-question plan (6 choices incl. 186 icons, 22 nouls),
  then small refine requests as text streams in. `npm run probe:jtbd` is its labeled benchmark.
- No hosted DiffusionGemma with logprobs exists (NVIDIA's trial NIM is undocumented on logprobs;
  OpenRouter has autoregressive Gemma 4 with logprobs).

**Prior art:** [open-jev](https://github.com/JoshuaSP/open-jev) (MIT) packs questions into one JSON
canvas with fixed structure tokens and reads allowed-token logits only at the final step. On an H100:
**134 ms at 1 step, 198 at 2, 330 at 4**; 1 step only became accurate after they stopped masking
during denoising and fixed the structure. Grouping cost 1–2 accuracy points. It is a Transformers
batch harness, not a server, and returns decisions, not probabilities. It ships TypeSafe's 20 public
eval cases (408 questions), a second benchmark source.

## 6. Dead ends (measured; don't retry without a new idea)

| Idea | Result |
|---|---|
| Gemini on Vertex for logprobs | Rejected by the owner; deleted. Newer Gemini models reject logprobs anyway. |
| vLLM `--safetensors-load-strategy=prefetch` | Slower: 18-minute load vs 14. The mount is the limit. |
| vLLM nightly for the concurrency fix | Crashes at warmup on this GPU; crash-looped and caused downtime. |
| Per-request step/entropy overrides | Ignored by the `gemma` image. |
| Questions-first prompt for cache hits | −60 ms, but −5/−6 on 40 labels. |
| Parallel isolated calls *on DiffusionGemma* | Blocked by the concurrency bug; even fixed, 12 calls ≈ 760 ms. (On an autoregressive model the same idea is the `scored` strategy and works.) |

## 7. Open decisions (the owner's)

**New, 2026-09-19, ahead of the older list (much of which the scored result makes moot: 1, 2, 6, 9):**

- **a. Go live with `scored`?** Which model: 26B-A4B (jev-level labels, parity on jtbd, 265 ms on plan)
  or E4B (faster than jev on both, weaker labels). Then retire `gev-model` (DiffusionGemma) or keep it.
- **b. Close the plan-suite gap on the 26B-A4B** (untried): FP8/NVFP4 weights; the icon question in one
  batch (raise `--max-logprobs` and use 186 single-token labels, or score option names as
  continuations); drop the repeated system text; run the API in the same container as vLLM.
- **c. Calibration**, now the main quality gap: temperature scaling per question type on half the
  labels (old item 7). Independent answers make this more promising than it was for packed.
- **d. Cold start and idle-out** still apply to any Cloud Run GPU service (old items 3, 8).

1. **How to reach ~1–2 denoising steps.**
   - **A. Custom model server (recommended):** Python + Transformers with open-jev's fixed canvas and
     final-logit readout, plus probabilities; replaces vLLM. The only route with published jev-range
     timings. Also fixes the icon question (one read of full-vocab logits at a slot instead of 13
     calls), sidesteps both vLLM bugs, and allows reading probabilities before the canvas hardens.
     Risks: their numbers are H100 + short prompts; ours is an RTX PRO 6000 + 1,900-token prompts;
     prefix caching must be hand-built; may land at 150–250 ms.
   - **B. Cheap probe first:** a *separate* vLLM service with a high `diffusion_entropy_bound` to see
     what 1–2 steps do to speed and to free-form sheet accuracy. ~40 GPU-minutes. Expectation: the
     free-form sheet breaks at low steps, which is why open-jev fixed the structure.
   - **C. Stop**, if 150–250 ms would not be good enough.
2. **Where to develop A.** Cloud Run costs 20 minutes per iteration. A GPU VM (or Modal, as open-jev
   used) with weights on local disk restarts in under a minute; port to Cloud Run once fast.
3. **Keep one model instance warm while working?** Avoids the 18-minute waits; costs a GPU around the
   clock until turned off. Price not yet checked.
4. **Redeploy the stale `gev` API** with current `main` (section 2).
5. **Default strategy** once latency is settled: packed (fast, jev-level accuracy, useless
   probabilities) or isolated (independent answers, slow).
6. **The icon question**, if not solved by A: shortlist by category then choose (2 calls), or make it
   a separate non-blocking request on the jev2ui side.
7. **Calibration:** temperature-scale logits on half the labeled set, test on the other half.
8. **Cold start**, untried: Direct VPC egress + Private Google Access; copy weights into RAM with
   parallel `gcloud storage cp` at startup (needs more memory); Run:ai streamer from `gs://`.
9. **The vLLM bug**, if staying on vLLM: force `TRITON_ATTN` on a nightly, wait for a rebuilt `gemma`
   image, or set `--max-num-seqs 1`. Packed mode makes it matter much less.

## 8. Suggested plan for option A

1. Get a GPU box with the weights on local disk. Reproduce open-jev's 1-step timing on *our* GPU with a
   jev2ui-sized prompt. **This is the go/no-go: if 1–2 steps isn't under ~150 ms server-side here,
   stop.**
2. Build the canvas from gev's question types: fixed `Q<n>:` structure, one answer slot per question;
   read the full-vocab logits at each slot from the final step; softmax over that question's allowed
   tokens. Try real values as tokens instead of letters (open-jev found letters weak in a canvas).
   Multi-token options need their trie approach; 186 icons need a check that names are distinct in
   their first token or a label scheme.
3. Add prefix/KV reuse for the constant question block (remember the prompt-order accuracy finding).
4. Serve it behind the same interface as `src/backends/vllm.ts` (`generate` → positions with
   top-k logprobs), or add a richer backend method that returns per-slot distributions directly.
5. Run `bench/compare.ts` on both suites; compare to the tables in section 4.
6. Port to Cloud Run; fix cold start separately.

## 9. Runbook

```bash
npm test && npm run typecheck

# Is the model warm? (a hang here means a ~18 min cold start is in progress)
export MODEL_URL=https://gev-model-huio5ftumq-uc.a.run.app TOKEN=$(gcloud auth print-identity-token)
curl -s -m 25 -o /dev/null -w "%{http_code}\n" $MODEL_URL/health -H "authorization: Bearer $TOKEN"

# Run gev locally against the live model (identity tokens last 1 hour)
GEV_STRATEGY=packed GEV_CONCURRENCY=1 GEV_MODEL_URL=$MODEL_URL/v1 GEV_MODEL_API_KEY=$TOKEN PORT=8787 node src/server.ts

# Benchmark a gev against recorded jev2ui traffic (suites: jtbd, plan) → bench/results/<suite>.<strategy>.json
GEV_URL=http://localhost:8787 GEV_API_KEY=x node bench/compare.ts jtbd [-v]
node --env-file=.env bench/compare.ts jtbd            # against the deployed API

# Model-server diagnostics
node bench/model-probe.ts latency | overrides | concurrency

# Record new fixtures from jev2ui (real jev calls, uses JEV_API_KEY)
node --env-file=.env bench/record.ts <suite>          # proxy on :8790
(cd ../jev2ui && TYPESAFE_BASE_URL=http://localhost:8790 npm run probe:jtbd)
(cd ../jev2ui && TYPESAFE_BASE_URL=http://localhost:8790 npx tsx ../gev/bench/capture-plan.ts)

# Point jev2ui at gev: TYPESAFE_BASE_URL=$GEV_URL JEV_API_KEY=$GEV_API_KEY

# Deploys (ask first). Model: ~20 min. API: ~2 min.
GOOGLE_CLOUD_PROJECT=gev-systemone ./scripts/deploy-model.sh
GOOGLE_CLOUD_PROJECT=gev-systemone ./scripts/deploy.sh

# State of the model service; clearing a failed newest revision = deploy a good one, then delete it
gcloud run revisions list --service gev-model --project gev-systemone --region us-central1
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="gev-model"' \
  --project gev-systemone --limit 30 --freshness 30m --format="value(timestamp,textPayload)"
```
