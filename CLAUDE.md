# gev — house rules

gev is a jev-compatible "System One" decision API (`POST /v1/systemone`) on GCP: DiffusionGemma
served by vLLM on a Cloud Run GPU, fronted by a small Node service. The consumer is
[jev2ui](https://github.com/dglazkov/jev2ui).

**Read [docs/STATE.md](docs/STATE.md) before doing anything.** It has the objective, what is deployed,
every measurement so far, the dead ends, the open decisions, and a runbook.

## The objective

Match jev's latency: about **150 ms from a client, ~110 ms server-side**, for a request of 18–28
questions. The owner's words: "If we can't, the point of this project is moot." Accuracy and
calibration matter, but after latency. Judge every idea by what it does to that number.

## Who decides what

The owner (Dimitri) makes the decisions. Ask first, with researched options and a recommendation, for:

- the model, the serving stack or image, the GPU type, the region
- anything that costs money (keeping instances warm, extra services, VMs, bigger machines)
- **any deploy to the live services** (`gev`, `gev-model`), including "obviously good" ones
- what to do when something fails: stop, report what you saw, give options. Do not switch to a
  fallback on your own, and do not ask for blanket permission to do so later.

Go ahead without asking on mechanical work that doesn't change what the owner is getting: bug fixes,
tests, benchmark tooling, scripts, docs, reverting your own failed experiment to the last known-good
state. If a live service is actively broken or burning money because of something you did, stop the
bleeding first, then report.

Don't substitute a convenient tool for the one the source material names. This project started with a
Gemini shortcut that had to be thrown away.

## Working agreements

- **Git:** commit and push straight to `main`. Two people, experimental, high velocity. No branches or
  PRs unless asked. No model identifiers in commits or code.
- **GCP project is `gev-systemone`.** The owner's gcloud default is a different project
  (`dandy-horse-3`) and must not be used or changed. Always pass `GOOGLE_CLOUD_PROJECT=gev-systemone`
  or `--project gev-systemone`.
- **Secrets live in `.env`** (gitignored, and excluded by `.gcloudignore` / `.dockerignore`):
  `JEV_API_KEY` (the real jev, for benchmarking), `GEMINI_API_KEY`, `GEV_URL`, `GEV_API_KEY`. Never
  print them, never put them in chat, logs, fixtures, or commits. Load with `node --env-file=.env`.
- **jev2ui is checked out at `../jev2ui`. Treat it as read-only.** Point it at another server with
  `TYPESAFE_BASE_URL`; never edit it.
- **The API contract is jev's.** jev2ui must work unmodified. Extra response fields are fine (`gev`);
  new required request fields are not. Strategy choices are server-side settings.
- **Measure, don't assume, and report negative results.** Several plausible ideas here measured worse
  (see Dead ends in STATE.md). Re-run the benchmark before and after any change to prompts or serving.
  When a measurement contradicts something you said earlier, say so.

## Hazards specific to this setup

- **A model-server deploy costs ~20 minutes** (48 GiB of weights through a slow mount) and can cause
  downtime. Never try a serving experiment on `gev-model`; use a separate service or a VM.
- **The warm model instance idles out after ~10 minutes.** The next request then waits ~18 minutes.
  Check `/health` before measuring, and don't read a hang as a bug.
- **Cloud Run keeps retrying a failed newest revision**, and each retry is a full GPU load. It also
  refuses to delete the newest revision. To clear one: deploy a good revision on top, then delete it.
- **The `vllm-openai:gemma` image corrupts concurrent logprobs requests** (HTTP 500, or another
  request's logprobs). Keep `GEV_CONCURRENCY=1` on it. Details in STATE.md.
- **Edit shell scripts with exact edits, not scripted string splices.** A splice keyed on a substring
  that also appeared in a comment once turned a usage example into a line that made
  `deploy-model.sh` call itself until the OS ran out of processes. After any script edit:
  `bash -n`, and check no uncommented line invokes the script itself.

## Code

- Node 24 runs the TypeScript directly: no build step, `.ts` import extensions, erasable syntax only
  (no enums, no parameter properties). Tests are `node:test`.
- `npm test && npm run typecheck` before every commit. `tsconfig` covers `src/` and `bench/`.
- Match the surrounding style: small modules, comments that say why, no speculative abstractions.
- Layout: `src/engine.ts` (strategies), `src/sheet.ts` (packed prompt + reader), `src/prompt.ts`
  (isolated prompts), `src/scoring.ts` (logprobs → distributions), `src/backends/vllm.ts` (the only
  backend), `bench/` (record / compare / model-probe), `scripts/` (deploys).
