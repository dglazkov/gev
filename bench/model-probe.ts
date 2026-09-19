// Talks to the model server directly (not through gev) to answer three questions about it:
//
//   latency      where a packed request's time goes, from vLLM's own timers: prefill vs denoising
//                steps vs cache hits, next to the client-observed time
//   overrides    whether per-request diffusion step/entropy overrides take effect
//   concurrency  whether concurrent logprobs requests fail (vLLM #57414)
//
//   MODEL_URL=https://gev-model-….run.app TOKEN=$(gcloud auth print-identity-token) \
//     node bench/model-probe.ts latency|overrides|concurrency [suite=jtbd]
//
// The model server is private, so TOKEN is a Google identity token of an account with run.invoker.
// A cold model server takes ~18 minutes to answer its first request; wait for /health first.

import { readFile } from "node:fs/promises";
import { SHEET_SYSTEM_INSTRUCTION, type SheetQuestion, sheetLabels, sheetMaxTokens, sheetPrompt } from "../src/sheet.ts";
import type { Question, SystemOneRequest } from "../src/types.ts";

const [mode = "latency", suite = "jtbd"] = process.argv.slice(2);
const { MODEL_URL, TOKEN, MODEL = "google/diffusiongemma-26B-A4B-it" } = process.env;
if (!MODEL_URL || !TOKEN) throw new Error("MODEL_URL and TOKEN are required");
const headers = { "content-type": "application/json", authorization: `Bearer ${TOKEN}` };

const COUNTERS = {
  steps: "vllm:diffusion_num_denoising_steps_total",
  cacheQueries: "vllm:prefix_cache_queries_total",
  cacheHits: "vllm:prefix_cache_hits_total",
  promptTokens: "vllm:prompt_tokens_total",
  generated: "vllm:generation_tokens_total",
  e2e: "vllm:e2e_request_latency_seconds_sum",
  requests: "vllm:e2e_request_latency_seconds_count",
  prefill: "vllm:request_prefill_time_seconds_sum",
  decode: "vllm:request_decode_time_seconds_sum",
  queue: "vllm:request_queue_time_seconds_sum",
} as const;
type Counters = Record<keyof typeof COUNTERS, number>;

async function counters(): Promise<Counters> {
  const text = await (await fetch(`${MODEL_URL}/metrics`, { headers })).text();
  const sum = (name: string) =>
    text.split("\n").filter((l) => l.startsWith(`${name}{`) || l.startsWith(`${name} `)).reduce((s, l) => s + Number(l.trim().split(" ").at(-1)), 0);
  return Object.fromEntries(Object.entries(COUNTERS).map(([key, name]) => [key, sum(name)])) as Counters;
}

async function fixture(n: number): Promise<SystemOneRequest> {
  const path = new URL(`./fixtures/${suite}/${String(n).padStart(3, "0")}.json`, import.meta.url);
  return JSON.parse(await readFile(path, "utf8")).request;
}

const sheetOf = (request: SystemOneRequest): SheetQuestion[] =>
  Object.entries(request.questions).flatMap(([id, question]: [string, Question]) => {
    const labels = sheetLabels(question);
    return labels ? [{ id, question, labels }] : [];
  });

async function ask(request: SystemOneRequest, extra: object = {}): Promise<{ status: number; answers: number; error?: string }> {
  const questions = sheetOf(request);
  const response = await fetch(`${MODEL_URL}/v1/chat/completions`, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: MODEL,
      messages: [
        { role: "system", content: SHEET_SYSTEM_INSTRUCTION },
        { role: "user", content: sheetPrompt(request.state, questions) },
      ],
      temperature: 0,
      max_tokens: sheetMaxTokens(questions.length) + 6,
      logprobs: true,
      top_logprobs: 20,
      chat_template_kwargs: { enable_thinking: false },
      ...extra,
    }),
  });
  if (!response.ok) return { status: response.status, answers: 0, error: (await response.text()).slice(0, 160) };
  const data = (await response.json()) as { choices?: { message?: { content?: string } }[] };
  return { status: 200, answers: (data.choices?.[0]?.message?.content?.match(/Q\d+: \S+/g) ?? []).length };
}

/** Runs `work` and reports what vLLM's counters say happened during it, per request. */
async function measured(label: string, work: () => Promise<{ clientMs: number[]; answers: number; note?: string }>) {
  const before = await counters();
  const { clientMs, answers, note } = await work();
  const after = await counters();
  const d = (k: keyof Counters) => after[k] - before[k];
  const n = Math.max(1, d("requests"));
  clientMs.sort((a, b) => a - b);
  const ms = (k: keyof Counters) => ((1000 * d(k)) / n).toFixed(0);
  console.log(`\n${label}${note ? `  [${note}]` : ""}`);
  console.log(`  client: median ${Math.round(clientMs[clientMs.length >> 1] ?? NaN)} ms (min ${Math.round(clientMs[0] ?? NaN)}, max ${Math.round(clientMs.at(-1) ?? NaN)})   answers read: ${answers}`);
  console.log(`  server per request: e2e ${ms("e2e")} ms = queue ${ms("queue")} + prefill ${ms("prefill")} + decode ${ms("decode")}`);
  console.log(`  denoising steps ${(d("steps") / n).toFixed(1)}   prompt tokens ${(d("promptTokens") / n).toFixed(0)}   prefix-cache hit ${((100 * d("cacheHits")) / Math.max(1, d("cacheQueries"))).toFixed(0)}%   generated ${(d("generated") / n).toFixed(0)} tokens`);
}

async function serial(requests: SystemOneRequest[], extra: object = {}) {
  const clientMs: number[] = [];
  let answers = 0;
  let note: string | undefined;
  for (const request of requests) {
    const started = performance.now();
    const result = await ask(request, extra);
    clientMs.push(performance.now() - started);
    answers += result.answers;
    if (result.error) note = `HTTP ${result.status} ${result.error}`;
  }
  return { clientMs, answers, note };
}

if (mode === "latency") {
  // Note that vLLM reports all diffusion work as "prefill"; steps × ~45 ms is the denoising share.
  const warm = await Promise.all([1, 2, 3, 4].map(fixture));
  const fresh = await Promise.all([5, 6, 7, 8, 9, 10, 11, 12].map(fixture));
  await measured("warmup (ignore: includes JIT and cold caches)", () => serial(warm));
  await measured("new states, shared question block (what production looks like)", () => serial(fresh));
  await measured("same requests again (prompt fully cached: the floor for prefill)", () => serial(fresh));
} else if (mode === "overrides") {
  // As of the June `vllm-openai:gemma` image, none of these change the step count.
  const requests = await Promise.all([21, 22, 23, 24].map(fixture));
  for (const [label, extra] of [
    ["baseline", {}],
    ["vllm_xargs.max_denoising_steps=1", { vllm_xargs: { max_denoising_steps: 1 } }],
    ["vllm_xargs.diffusion_max_steps=1", { vllm_xargs: { diffusion_max_steps: 1 } }],
    ["vllm_xargs.diffusion_entropy_bound=10", { vllm_xargs: { diffusion_entropy_bound: 10 } }],
    ["top-level max_denoising_steps=1", { max_denoising_steps: 1 }],
  ] as [string, object][]) {
    await measured(label, () => serial(requests, extra));
  }
} else if (mode === "concurrency") {
  // Before vLLM #57414, a share of these fail with HTTP 500 "list index out of range".
  const request = await fixture(2);
  for (const n of [1, 4, 12]) {
    const started = performance.now();
    const results = await Promise.all(Array.from({ length: n }, () => ask(request)));
    const failed = results.filter((r) => r.status !== 200);
    console.log(`${n} concurrent: ${failed.length} failed${failed[0] ? ` (HTTP ${failed[0].status} ${failed[0].error})` : ""}, ${Math.round(performance.now() - started)} ms`);
  }
} else {
  throw new Error(`unknown mode "${mode}" (expected latency, overrides, or concurrency)`);
}
