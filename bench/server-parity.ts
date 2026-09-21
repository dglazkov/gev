// Compares what gev asks an SGLang server with what it asks a vLLM server serving the same model,
// token by token, and the answers it gets back. A prompt that is off by a token shifts every answer
// and nothing else would show it: the requests still succeed. (Expect one difference: vLLM's chat
// template puts a space after the system text and SGLang's doesn't; see docs/STATE.md.)
//
//   VLLM_URL=https://gev-ar-fp8-….run.app SGLANG_URL=https://gev-sglang-….run.app \
//     MODEL=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic TOKEN=$(gcloud auth print-identity-token) \
//     node bench/server-parity.ts [suite=plan]
//
// Uses the live settings (scored, state-last, wide choice, T=4). Compares, for every fixture, the
// token ids of every prompt as each server tokenizes it, the first tokens of every option name,
// and the answers.

import { readFile, readdir } from "node:fs/promises";
import { type ModelServer, VllmBackend } from "../src/backends/vllm.ts";
import { DEFAULT_ENGINE_OPTIONS, type EngineOptions, systemOne } from "../src/engine.ts";
import type { Answer, SystemOneRequest, SystemOneResponse } from "../src/types.ts";

const [suite = "plan"] = process.argv.slice(2);
const { VLLM_URL, SGLANG_URL, TOKEN, MODEL = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic" } = process.env;
if (!VLLM_URL || !SGLANG_URL || !TOKEN) throw new Error("VLLM_URL, SGLANG_URL and TOKEN are required");
const headers = { "content-type": "application/json", authorization: `Bearer ${TOKEN}` };
const options: EngineOptions = { ...DEFAULT_ENGINE_OPTIONS, strategy: "scored", order: "state-last", wideChoice: true, temperature: { choice: 4, score: 4, noul: 1 } };

const dir = new URL(`./fixtures/${suite}/`, import.meta.url);
const requests: SystemOneRequest[] = await Promise.all(
  (await readdir(dir)).filter((n) => n.endsWith(".json")).sort().map(async (n) => JSON.parse(await readFile(new URL(n, dir), "utf8")).request),
);

// The completions bodies each backend sends, captured on their way out.
const sent = new Map<string, any[]>();
const realFetch = globalThis.fetch;
globalThis.fetch = (async (url: string, init?: RequestInit) => {
  const { origin, pathname } = new URL(url);
  if (pathname === "/v1/completions") sent.set(origin, [...(sent.get(origin) ?? []), JSON.parse(String(init?.body))]);
  return realFetch(url, init);
}) as typeof fetch;

const servers: [ModelServer, string][] = [["vllm", VLLM_URL], ["sglang", SGLANG_URL]];
const backends = servers.map(([server, url]) => new VllmBackend({ baseUrl: `${url}/v1`, model: MODEL, apiKey: TOKEN, server }));

/** What the server itself makes of a prompt: the ids it will run, with <bos> added only where gev relies on it. */
async function ids(server: ModelServer, url: string, prompt: string): Promise<number[]> {
  const body = { model: MODEL, prompt, add_special_tokens: server === "sglang" };
  const response = await realFetch(`${url}/tokenize`, { method: "POST", headers, body: JSON.stringify(body) });
  return (await response.json()).tokens;
}

const names = (request: SystemOneRequest) =>
  Object.values(request.questions).flatMap((q) => (q.type === "choice" ? [Object.keys(q.criteria)] : []));

let prompts = 0;
let promptsDiffer = 0;
const diffs = new Map<string, number>();
let nameListsDiffer = 0;
const verdicts = { same: 0, differ: 0 };
let maxGap = 0;
const differ: string[] = [];

for (const [n, request] of requests.entries()) {
  for (const list of names(request)) {
    const [a, b] = await Promise.all(backends.map((backend) => backend.firstTokens!(list)));
    if (JSON.stringify(a) !== JSON.stringify(b)) nameListsDiffer++;
  }
  sent.clear();
  const [a, b] = (await Promise.all(backends.map((backend) => systemOne(backend, request, options)))) as [SystemOneResponse, SystemOneResponse];
  // Same prompts, in the same order, tokenized by their own servers.
  const [vllmBodies, sglangBodies] = servers.map(([, url]) => (sent.get(new URL(url).origin) ?? []).flatMap((body) => body.prompt as string[]));
  for (const [i, prompt] of (vllmBodies ?? []).entries()) {
    prompts++;
    const [x, y] = await Promise.all([ids("vllm", VLLM_URL, prompt), ids("sglang", SGLANG_URL, sglangBodies![i]!)]);
    if (JSON.stringify(x) !== JSON.stringify(y)) {
      promptsDiffer++;
      // Where they differ: strip the common start and end, and tally what is left on each side.
      let start = 0;
      while (start < Math.min(x.length, y.length) && x[start] === y[start]) start++;
      let end = 0;
      while (end < Math.min(x.length, y.length) - start && x[x.length - 1 - end] === y[y.length - 1 - end]) end++;
      const kind = `at token ${start}: vLLM [${x.slice(start, x.length - end)}] vs SGLang [${y.slice(start, y.length - end)}]`;
      diffs.set(kind, (diffs.get(kind) ?? 0) + 1);
    }
  }
  for (const [key, ours] of Object.entries(a.answers)) {
    const theirs: Answer = b.answers[key]!;
    const decision = (x: Answer) => (x.type === "choice" ? x.choice : x.type === "score" ? Math.round(x.score) : x.noul >= 0.5);
    const gap = ours.type === "noul" && theirs.type === "noul" ? Math.abs(ours.noul - theirs.noul) : ours.type === "choice" && theirs.type === "choice" ? Math.abs(ours.confidence - theirs.confidence) : ours.type === "score" && theirs.type === "score" ? Math.abs(ours.score - theirs.score) : NaN;
    maxGap = Math.max(maxGap, gap);
    if (decision(ours) === decision(theirs)) verdicts.same++;
    else {
      verdicts.differ++;
      differ.push(`${n + 1}.${key}: ${decision(ours)} vs ${decision(theirs)}`);
    }
  }
}

console.log(`${suite}: ${requests.length} requests`);
console.log(`  prompts token-identical: ${prompts - promptsDiffer}/${prompts}`);
for (const [kind, count] of diffs) console.log(`    ${count} differ ${kind}`);
console.log(`  option-name lists with the same first tokens: ${nameListsDiffer === 0 ? "all" : `all but ${nameListsDiffer}`}`);
console.log(`  same decision: ${verdicts.same}/${verdicts.same + verdicts.differ}   largest gap (noul p / choice confidence / score): ${maxGap.toFixed(3)}`);
if (differ.length) console.log(`  differ: ${differ.join(", ")}`);
