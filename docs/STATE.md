# State of gev

Last updated 2026-09-20. Keep this current: update it in the same commit as any change that makes
part of it untrue. House rules are in [CLAUDE.md](../CLAUDE.md).

## 1. Objective

Match [jev](https://docs.typesafe.ai/)'s latency with an open model on GCP.

| from a laptop, median / p90 | jev | **gev live** (FP8 Gemma 4 26B-A4B, `scored`) | gev on the morning of 2026-09-19 (DiffusionGemma, packed) |
|---|---|---|---|
| 18 questions (jtbd) | 151 / 255 ms | **132 / 182 ms** (127–132 / 136–182 across runs) | ~376 ms |
| 28 questions incl. 186 icons (plan) | 181 / 273 ms | **175 / 206 ms** (171–178 / 206–246 across runs) | ~3,300 ms |
| of which waiting on the model server | ~110 ms | ~65 ms (jtbd), ~80 ms (plan) | 220–310 ms |

**The latency objective is met and live since 2026-09-19**, from a laptop whose RTT to us-central1 (~58 ms)
is 17 ms worse than its RTT to jev. It was reached by dropping diffusion for one batched forward pass
of an autoregressive model (section 3, `scored`). Accuracy is at jev's level on the hand labels (109/120
vs jev's 103). Choices and scores are calibrated with a temperature; **noul probabilities are still
overconfident** (section 4). **Next thing to try: gev in the same container as vLLM** (section 7).

## 2. What is deployed

GCP project `gev-systemone` (org glazkov.com, Personal Billing), region `us-central1`.

| Service | What | Notes |
|---|---|---|
| `gev` | The API, `https://gev-huio5ftumq-uc.a.run.app` | Public, bearer key (`GEV_API_KEY` in `.env`; Secret Manager `gev-api-keys`), CORS open on `/v1/*`. Demo page at `/`. **Live settings** (the defaults of `scripts/deploy.sh`): model service `gev-ar-fp8`, `GEV_STRATEGY=scored`, `GEV_PROMPT_ORDER=state-last`, `GEV_WIDE_CHOICE=1`, `GEV_TEMPERATURE=choice=4;score=4`. |
| `gev-ar-fp8` | **The live model server** (the defaults of `scripts/deploy-model.sh`). vLLM `v0.29.0` + `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic`, private | RTX PRO 6000, **min 1 / max 1 instance: a GPU around the clock**, because a cold start takes ~10 minutes (section 5). `--max-num-seqs 128 --max-num-batched-tokens 16384 --max-logprobs 256`. A third-party FP8 quantization of Google's weights (Red Hat, the vLLM maintainers). **Never experiment on it.** |
| `gev-ar` | The same model in bf16 (Google's own weights), private, scale to zero | The fallback if FP8 is ever in doubt. Deployed with `--max-logprobs 20`: redeploy it with the current script before using `GEV_WIDE_CHOICE` on it. |
| `gev-scored` | Staging copy of the API, `https://gev-scored-huio5ftumq-uc.a.run.app` | Shares `gev`'s keys; same settings as live, pointed at the live model server. Try API-side changes here first: `SERVICE=gev-scored SECRET=gev-api-keys [MODEL_SERVICE=… MODEL=… EXTRA_ENV=…] ./scripts/deploy.sh`. |
| `gs://gev-systemone-models` | Weights | `RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic` (live), `google/gemma-4-26B-A4B-it`, `google/gemma-4-E4B-it`; mounted read-only at `/models`. DiffusionGemma and NVFP4 weights were deleted 2026-09-20; `deploy-model.sh` re-copies any model from Hugging Face on demand. |

GPU quota in us-central1 is 3 RTX PRO 6000s across all services; a fourth instance fails to deploy with
"Quota exceeded for total allowable count of GPUs". Idle services at zero instances don't count.

Service accounts: `gev-runtime@` and `gev-scored-runtime@` (may only invoke the model services and read
the key secret), `gev-ar-fp8@` and `gev-ar@` (may only read the bucket).

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

**Round two on `scored` (2026-09-19, later).** All through `gev-scored` from the laptop unless noted.

| | jtbd median / p90 | plan median / p90 | hand labels (stage, done, highStakes) | agrees with jev, jtbd / plan |
|---|---|---|---|---|
| jev | 151 / 255 | 181 / 273 | 36, 29, 38 = 103 | |
| 26B-A4B bf16, tournament | 146 / 156 | 264 / 328 | 38, 29, 39 = 106 | 78% / 85% |
| **26B-A4B NVFP4 (RedHatAI), icons by name** | **119 / 140** | **206 / 251** | 34, 31, 38 = 103 | **74%** / 86% |

- **Where the time goes** (`gev.ms`, `gev.model_ms` in every response; `compare.ts` prints the split).
  bf16: jtbd 146 = 64 outside gev + 1 gev + 80 model wait; plan 264 = 82 + 3 + 178 (two calls).
  NVFP4 + by-name: jtbd 119 = 62 + 1 + 56; plan 206 = 80 + 1 + 126. "Outside gev" is the laptop's
  ~58 ms RTT plus a little; gev's own work is nothing; vLLM's e2e is ~20 ms less than the model wait.
- **NVFP4 costs accuracy on jtbd**: choice agreement with jev 81% → 71%, overall 78% → 74%; hand labels
  103 vs 106 (within noise). Speed gain is modest (vLLM e2e 58 → 50 ms).
- **FP8 (RedHatAI FP8-dynamic) is the best of the three**: hand labels 38, 31, 40 = 109; agreement with
  jev 78% (jtbd) and 86% (plan), same as bf16; vLLM e2e 45 ms on jtbd. Through the API: jtbd 125 / 133,
  plan 198 / 237. This is what went live.
- **Calibration** (`GEV_TEMPERATURE`, `bench/calibrate.ts`, FP8, both suites). jev's sharpness (0 =
  undecided, 1 = certain): choice 0.74, score 0.62, noul 0.55; gev at T=1: 0.98, 0.92, 0.99. For
  choices T≈4 is a real optimum (distance to jev 0.272 → 0.239 on the held-out half, sharpness 0.68)
  and for scores T≈4 matches jev's sharpness (0.65; distance 0.142 → 0.120). **Nouls don't calibrate
  with one temperature**: distance keeps falling to T=8+ only because hedging beats disagreeing, and
  sharpness is still 0.76 at T=8; agreement with jev on the 0.5 / 0.6 thresholds stays 84% / 89%
  whatever T is. Not applied live; owner's call.
- **The icon question by name** (`GEV_WIDE_CHOICE=1`): the model answers with the icon's name and gev
  reads the first token with `logprobs` = names + 40 (server needs `--max-logprobs=256`). 163 of the 186
  names have a first token of their own; the 9 sets that share one (`shopping_cart`/`shopping_bag`,
  six `local_*`) are split by lettered questions in the same batch. One round instead of two; picks
  are as close to jev's as the tournament's (7/13 each) and sensible where they differ.
- **Model-side time scales with the number of prompts in the batch**, ~2.5–3 ms per prompt even when
  everything but the last block is cached: 1 prompt ≈ 17 ms, 18 ≈ 56–70, 27 ≈ 95, 36 ≈ 116
  (NVFP4, laptop time minus RTT). Plan is slow because it is 37 prompts, not because of the icons.

**Round three (2026-09-19, night): fewer prompts, serving flags.** Code in `main`; on `gev-scored`, **not
yet on live `gev`** (owner's go-ahead needed).

| laptop → API → model, median / p90 | jtbd | plan |
|---|---|---|
| jev | 151 / 255 | 181 / 273 |
| live `gev` (round two code) | 127 / 141 | 194 / 248 |
| **round three code, same FP8 model server** | 129 / 136 | **171 / 243** |

- **vLLM's cost is per prompt, not per token**: 9 / 18 / 36 / 72 prompts = 104 / 124 / 156 / 226 ms from
  the laptop (≈1.9 ms per prompt, of which ≈0.5 is its ~40 uncached tokens); +2,000 state tokens
  across 18 prompts costs only +23 ms. So: fewer prompts and less work per prompt; shortening the text
  after the state is not worth doing.
- **Rotated names**: an option name that shares its first token with another is shown with its words
  rotated until it starts with a token of its own (`shopping_cart` → `cart_shopping`), which removes
  the lettered sub-questions: plan goes from 37 prompts to 28. Icon agreement with jev 8/13 (tournament 6–7).
- **Nouls read 5 logprobs, not 20**, in their own call alongside the rest.
- Agreement with jev unchanged by both (jtbd 78%, plan 85–86%).
- **`--max-cudagraph-capture-size=4096`**: helps NVFP4 (GPU step 36 → 23 ms on jtbd; laptop 129 → 119,
  plan 176 → 160) but makes **no measurable difference on FP8** through the API (jtbd 125 vs 129, plan
  178 vs 171). Live server left as is.
- In-region, a model call takes ~60 ms where vLLM's own e2e is ~37: ~23 ms is Cloud Run's front end,
  auth, and vLLM's HTTP layer. Running gev in the same container as vLLM (untried) could recover some.
- **The live model server idles out after ~10–15 minutes and takes ~10 minutes to come back** (447 s of
  weights + 107 s of init). Requests that arrive meanwhile hang, and when several queue up Cloud Run
  answers 429. This happened repeatedly during benchmarking and is now the biggest practical
  latency problem. `--min-instances 1` fixes it at the cost of a GPU around the clock (owner's call).

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
- **Cold start ≈ 10 minutes for the 26 GiB FP8 weights** (≈ 18 for 48 GiB of bf16): the Cloud Storage
  FUSE mount delivers ~50–60 MB/s regardless of read parallelism, and two services loading at once
  share it. Cloud Build moved the same bytes HF → bucket in 4 minutes.
- Idle instances are reclaimed after ~10–15 minutes unless `--min-instances 1` (the live server has it).
  Requests that arrive during a load hang, and those queued behind them get 429s. Experiment services
  at min 0 idle out between measurements: check `/health` first.
- A failed newest revision is retried indefinitely (a full GPU load each time) and cannot be deleted
  until a newer revision exists.
- `gcloud run deploy --source` uploads everything not in `.gcloudignore`, including `.env` if absent.
- Cloud Run's front end keeps `/healthz` for itself: gev's `/healthz` answers locally but returns a
  Google 404 when deployed. To check the live API, send a real request (or open `/`).

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
| 186 icons under two-letter single-token labels in one prompt | As close to jev as the tournament (6/13) but the misses were nonsense (`location_on` for an energy dashboard): arbitrary labels bind poorly at that length. Answer-by-name replaced it. |
| One completions call read as wide as its widest prompt (logprobs 212 for all 37) | Plan 308 ms vs 176 with two calls: vLLM's wide-logprobs cost is per prompt. |
| Logprob widths graded by label count (5 / 10 / 20) instead of nouls-only | jtbd 102 vs 104 ms, plan 132 vs 122: each extra width is another call. Reverted. |
| `--moe-backend=flashinfer_trtllm` / `flashinfer_cutlass` for the FP8 checkpoint | Both refuse to start on this setup: TRTLLM "does not support current device", CUTLASS doesn't support the checkpoint's per-channel × per-token FP8 scheme. vLLM's pick, untuned Triton ("Using default MoE config"), is the only one; tuning it for this GPU is untried. Fails before loading weights, so cheap to find out. |
| Streaming weights from the bucket (`--load-format=runai_streamer`, `MODEL_PATH=gs://…`) to shorten the cold start | Slower: 554 s vs 447 s through the mount for the 26 GiB FP8 weights. Works out of the box in the `v0.29.0` image once the service account has `storage.buckets.get` (`roles/storage.legacyBucketReader`). The limit is the container's network path to Cloud Storage (~50–60 MB/s) however the bytes are read; the untried fix is still Direct VPC egress + Private Google Access, or keeping an instance warm. |
| Shortening the prompt after the state | Not tried, because measured to be pointless: see "per prompt, not per token". |
| Sending token ids instead of text to skip vLLM's tokenizer | 151 vs 155 ms: tokenization is not the cost. `logprobs: 20` costs ~9 ms over none. |
| Parallel isolated calls *on DiffusionGemma* | Blocked by the concurrency bug; even fixed, 12 calls ≈ 760 ms. (On an autoregressive model the same idea is the `scored` strategy and works.) |

## 7. Next, and open decisions

Decided by the owner on 2026-09-19/20 and done: go live on the FP8 26B-A4B with `scored`; round three
(rotated names, 5 logprobs for nouls) live; `GEV_TEMPERATURE=choice=4;score=4` live; keep the live
model server warm; delete `gev-model`, `gev-ar-e4b`, `gev-ar-fp8x`, `gev-ar-nvfp4` and the DiffusionGemma
and NVFP4 weights.

**Next thing to try (owner, 2026-09-20): run gev in the same container as vLLM.** In-region a model call
takes ~60–65 ms where vLLM's own e2e is ~37–45 ms; the ~20 ms between is Cloud Run's front end, the ID
token check, TLS and vLLM's HTTP layer, paid once per call (jtbd 2 parallel calls, plan 3). Sketch:
one Cloud Run service with two containers, gev as the ingress container and vLLM as a sidecar holding
the GPU, talking over localhost with no auth (`GEV_MODEL_URL=http://localhost:8000/v1`). Things to
settle: the service becomes public *and* holds the GPU (scaling, min 1, the 20 vCPU / 80 GiB minimum
is per service), startup ordering (gev must report healthy only when vLLM is), API deploys would then
restart vLLM unless the revision keeps the instance, which it won't, so **every API deploy becomes a
~10-minute model load**; a staging twin needs a second GPU. Build it as a separate service, measure with
`bench/compare.ts` (the `model_ms` split shows the gain directly), and only then ask to switch.
Expected gain: 10–20 ms per request, unmeasured.

Other open items:

1. **Noul calibration.** One temperature doesn't do it (section 4). Needs labeled nouls, or a different
   idea (per-question bias terms fitted against jev, asking both polarities).
2. **Tune the Triton FP8 MoE kernel for this GPU**: vLLM warns it runs a default config
   (`E=128,N=704,…RTX_PRO_6000…fp8_w8a8.json` not found). vLLM ships a tuning script; needs its own GPU
   service for an hour or so. Gain unknown.
3. **GPU quota is 3**; with the live server holding one permanently, two experiments at most. Ask for more
   before the next round of serving experiments.
4. **Cold start** still ~10 minutes whenever the live revision is replaced or crashes. Untried: Direct VPC
   egress + Private Google Access for the bucket path.
5. **What the warm GPU costs** has not been looked up.
6. `gev-ar` (bf16 fallback) and the E4B weights are kept; delete when no longer wanted.
7. **The API itself still scales to zero**: the first request after an idle spell takes ~1.8 s (Node
   starting, fetching an ID token, learning the chat template) instead of ~130 ms. `--min-instances 1`
   on `gev` is a small CPU-only standing cost; not applied, owner's call.

## 8. History of the approach (for context; superseded)

Until the afternoon of 2026-09-19 the plan was to make DiffusionGemma fast: a packed answer sheet on
vLLM (~376 ms), then a custom fixed-canvas server after [open-jev](https://github.com/JoshuaSP/open-jev)
to get to 1–2 denoising steps. The observation that ended it: the state is ~15 tokens and the questions
~1,900 and constant per app, and at one denoising step a diffusion model does nothing that a single
autoregressive forward pass doesn't. The DiffusionGemma measurements and learnings in sections 4–6 are
kept because they are true, not because they are the way forward.

## 9. Runbook

```bash
npm test && npm run typecheck

# Is the model up? (kept warm; a hang here means a ~10 min load is in progress after a redeploy or crash)
export MODEL_URL=https://gev-ar-fp8-huio5ftumq-uc.a.run.app TOKEN=$(gcloud auth print-identity-token)
curl -s -m 25 -o /dev/null -w "%{http_code}\n" $MODEL_URL/health -H "authorization: Bearer $TOKEN"

# Run gev locally against the live model (identity tokens last 1 hour)
GEV_STRATEGY=scored GEV_PROMPT_ORDER=state-last GEV_WIDE_CHOICE=1 GEV_TEMPERATURE="choice=4;score=4" \
  GEV_MODEL_URL=$MODEL_URL/v1 GEV_MODEL_API_KEY=$TOKEN PORT=8787 node src/server.ts

# Benchmark a gev against recorded jev2ui traffic (suites: jtbd, plan) → bench/results/<suite>.<strategy>.json
GEV_URL=http://localhost:8787 GEV_API_KEY=x node bench/compare.ts jtbd [-v]
node --env-file=.env bench/compare.ts jtbd            # against the deployed API

# Model-server diagnostics (MODEL=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic for the live server)
ORDERS=state-last WIDE=1 node bench/model-probe.ts score jtbd|plan   # accuracy, client ms, vLLM's own timers
node bench/calibrate.ts jtbd plan --temperatures 1,2,3,4,5,6         # GEV_TEMPERATURE sweep against jev
node bench/model-probe.ts latency | overrides | concurrency          # DiffusionGemma-era probes

# Record new fixtures from jev2ui (real jev calls, uses JEV_API_KEY)
node --env-file=.env bench/record.ts <suite>          # proxy on :8790
(cd ../jev2ui && TYPESAFE_BASE_URL=http://localhost:8790 npm run probe:jtbd)
(cd ../jev2ui && TYPESAFE_BASE_URL=http://localhost:8790 npx tsx ../gev/bench/capture-plan.ts)

# Point jev2ui at gev: TYPESAFE_BASE_URL=$GEV_URL JEV_API_KEY=$GEV_API_KEY

# Deploys (ask first). Defaults are the live services. Model: ~10 min. API: ~2 min.
GOOGLE_CLOUD_PROJECT=gev-systemone ./scripts/deploy-model.sh
GOOGLE_CLOUD_PROJECT=gev-systemone ./scripts/deploy.sh

# State of the model service; clearing a failed newest revision = deploy a good one, then delete it
gcloud run revisions list --service gev-ar-fp8 --project gev-systemone --region us-central1
gcloud logging read 'resource.type="cloud_run_revision" AND resource.labels.service_name="gev-ar-fp8"' \
  --project gev-systemone --limit 30 --freshness 30m --format="value(timestamp,textPayload)"
```
