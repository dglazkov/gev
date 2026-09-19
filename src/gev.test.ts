import assert from "node:assert/strict";
import { test } from "node:test";
import { createApp } from "./app.ts";
import { type Backend, type Generation, type Position, type Scores, UpstreamError } from "./backends/backend.ts";
import { answerPositions, chatFrame, framed } from "./backends/vllm.ts";
import { DEFAULT_ENGINE_OPTIONS, systemOne } from "./engine.ts";
import { engineOptionsFromEnv } from "./config.ts";
import { confidence, expectedLevel, labelDistribution, temper } from "./scoring.ts";
import { PROMPT_ORDERS, noulPrompt } from "./prompt.ts";
import { readSheet } from "./sheet.ts";
import type { SystemOneRequest, SystemOneResponse } from "./types.ts";

const near = (actual: number, expected: number, eps = 1e-3) =>
  assert.ok(Math.abs(actual - expected) < eps, `expected ${actual} ≈ ${expected}`);

/**
 * Puts `p` of the mass on whichever label sits next to `favorite` (else the first label).
 * Understands both prompt shapes: a single question, and a sheet of "Q<n>" questions.
 */
class FakeBackend implements Backend {
  readonly model = "fake-1";
  readonly prompts: string[] = [];
  /** Sheet question numbers to leave off the reply, as a model that loses its place would. */
  skip = new Set<number>();
  protected readonly favorite: string;
  readonly #p: number;
  inflight = 0;
  peak = 0;

  constructor(favorite: string, p = 0.9) {
    this.favorite = favorite;
    this.#p = p;
  }

  #position(labels: string[], block: string): Position {
    const line = block.split("\n").find((l) => /^\w+: /.test(l) && l.includes(this.favorite));
    const favorite = line?.split(":")[0] ?? labels[0]!;
    const rest = (1 - this.#p) / (labels.length - 1);
    return { token: ` ${favorite}`, top: labels.map((label) => ({ token: ` ${label}`, logprob: Math.log(label === favorite ? this.#p : rest) })) };
  }

  async generate(_system: string, prompt: string): Promise<Generation> {
    this.prompts.push(prompt);
    this.peak = Math.max(this.peak, ++this.inflight);
    await new Promise((resolve) => setImmediate(resolve));
    this.inflight--;
    const usage = { input_tokens: 10, output_tokens: 1 };
    const plain = (token: string): Position => ({ token, top: [{ token, logprob: 0 }] });

    if (!prompt.includes("\nQUESTIONS:\n")) {
      return { positions: [this.#position(replyLabels(prompt), prompt)], usage };
    }
    const blocks = prompt.split("\nQUESTIONS:\n")[1]!.split(/\n+Q\d+\. /).slice(1);
    const template = prompt.split("\n").filter((l) => /^Q\d+: /.test(l));
    const positions = template.flatMap((line, i) => {
      if (this.skip.has(i + 1)) return [];
      const labels = line.replace(/^Q\d+: /, "").split(" | ");
      // Digits of the question number arrive as separate tokens, as Gemma's tokenizer emits them.
      return [plain("Q"), ...String(i + 1).split("").map(plain), plain(":"), this.#position(labels, blocks[i]!), plain("\n")];
    });
    return { positions, usage };
  }
}

/** The `gev` block without its timings, which vary from run to run. */
const untimed = (gev: SystemOneResponse["gev"]) => {
  const { ms: _ms, model_ms: _modelMs, ...rest } = gev!;
  return rest;
};

const replyLabels = (prompt: string) =>
  prompt.split("\n").find((l) => l.startsWith("Reply with exactly one of: "))!.replace("Reply with exactly one of: ", "").split(", ");

/** A FakeBackend that can also score a batch of isolated prompts, as an autoregressive model can. */
class FakeScoringBackend extends FakeBackend {
  readonly batches: string[][] = [];
  readonly tops: (number | undefined)[] = [];

  /** A tokenizer that splits after the first digit: "option-37" starts with "option-3", as do nine others. */
  async firstTokens(strings: string[]): Promise<string[]> {
    return strings.map((s) => /^\D*\d?/.exec(s)![0]);
  }

  async score(system: string, prompts: string[], top?: number): Promise<Scores> {
    this.batches.push(prompts);
    this.tops.push(top);
    if (prompts[0]!.includes("copied exactly")) {
      // Answering by name: all the mass that isn't on the favorite's first token goes to some other name's.
      const names = prompts[0]!.split("ANSWERS:\n")[1]!.split("\n\n")[0]!.split("\n");
      const firsts = [...new Set(await this.firstTokens(names))];
      const favorite = (await this.firstTokens([names.find((n) => n.includes(this.favorite))!]))[0]!;
      return { tops: [firsts.map((token) => ({ token, logprob: Math.log(token === favorite ? 0.9 : 0.1 / (firsts.length - 1)) }))], usage: { input_tokens: 10, output_tokens: 1 } };
    }
    const generations = await Promise.all(prompts.map((prompt) => this.generate(system, prompt)));
    return { tops: generations.map((g) => g.positions[0]!.top), usage: { input_tokens: 10 * prompts.length, output_tokens: prompts.length } };
  }
}

test("labelDistribution pools token variants, drops off-script tokens, floors unseen labels", () => {
  const p = labelDistribution(
    [
      { token: "A", logprob: Math.log(0.5) },
      { token: " a", logprob: Math.log(0.1) },
      { token: "▁B", logprob: Math.log(0.2) },
      { token: "The", logprob: Math.log(0.1) },
    ],
    ["A", "B", "C"],
  );
  // C is unseen: floor = 0.1 / 2. Total = 0.6 + 0.2 + 0.05.
  near(p[0]!, 0.6 / 0.85);
  near(p[1]!, 0.2 / 0.85);
  near(p[2]!, 0.05 / 0.85);
});

test("labelDistribution is uniform when the backend returns nothing", () => {
  assert.deepEqual(labelDistribution([], ["yes", "no"]), [0.5, 0.5]);
});

test("answerPositions drops Gemma's empty thought channel", () => {
  const at = (token: string) => ({ token, top_logprobs: [{ token, logprob: 0 }] });
  const tokens = (names: string[]) => answerPositions(names.map(at)).map((p) => p.token);
  // As observed from DiffusionGemma with thinking disabled.
  assert.deepEqual(tokens(["<|channel>", "thought", "\n", "<channel|>", "A", "<turn|>"]), ["A", "<turn|>"]);
  // About a third of the time it leaves the channel unclosed and goes straight to the answer.
  assert.deepEqual(tokens(["<|channel>", "thought", "\n", "Q", "1", ":", " F", "\n"]), ["Q", "1", ":", " F", "\n"]);
  assert.deepEqual(tokens(["B", "<turn|>"]), ["B", "<turn|>"]);
  assert.deepEqual(tokens(["<|channel>", "thought"]), []);
});

test("readSheet finds answers by line, whatever the tokenization, and ignores off-format lines", () => {
  const at = (token: string, top: [string, number][] = [[token, 1]]): Position => ({ token, top: top.map(([t, p]) => ({ token: t, logprob: Math.log(p) })) });
  const questions = [
    { id: "team", labels: ["A", "B"] },
    { id: "urgent", labels: ["yes", "no"] },
    { id: "skipped", labels: ["yes", "no"] },
    { id: "bogus", labels: ["0", "1", "2"] },
  ].map((q) => ({ ...q, question: { type: "noul", instructions: "" } as const }));
  const read = readSheet(
    [
      at("Q1"), at(":"), at(" B", [[" B", 0.8], [" A", 0.2]]), at("\n"),
      at("Q"), at("2"), at(":"), at(" "), at("no", [["no", 0.7], ["yes", 0.3]]), at("\n"),
      at("Q4: "), at("maybe"), at("\n"),
    ],
    questions,
  );
  assert.deepEqual([...read.keys()], ["team", "urgent"]);
  near(read.get("team")![1]!, 0.8);
  near(read.get("urgent")![0]!, 0.3);
});

test("temperature softens probabilities without changing the decision", async () => {
  near(temper([0.99, 0.01], 2)[0]!, Math.sqrt(0.99) / (Math.sqrt(0.99) + Math.sqrt(0.01)));
  assert.deepEqual(temper([0.7, 0.3], 1), [0.7, 0.3]);
  const request: SystemOneRequest = { state: "s", questions: { flag: { type: "noul", instructions: "it is raining" } } };
  const plain = await systemOne(new FakeBackend("no such line", 0.99), request);
  const soft = await systemOne(new FakeBackend("no such line", 0.99), request, { ...DEFAULT_ENGINE_OPTIONS, temperature: { choice: 1, score: 1, noul: 3 } });
  assert.ok(plain.answers.flag?.type === "noul" && soft.answers.flag?.type === "noul");
  assert.ok(soft.answers.flag.noul < plain.answers.flag.noul && soft.answers.flag.noul > 0.5);
  assert.deepEqual(engineOptionsFromEnv({ GEV_TEMPERATURE: "2,noul=3" }).temperature, { choice: 2, score: 2, noul: 3 });
  assert.throws(() => engineOptionsFromEnv({ GEV_TEMPERATURE: "maybe=2" }), /GEV_TEMPERATURE/);
});

test("confidence and expectedLevel", () => {
  near(confidence([0.25, 0.25, 0.25, 0.25]), 0);
  near(confidence([1, 0, 0]), 1);
  near(expectedLevel([0.05, 0.3, 0.65]), 1.6);
});

test("answers all three question types in the jev response shape", async () => {
  const backend = new FakeBackend("technical");
  const response = await systemOne(backend, {
    state: { document: "The API returns 500 on every call." },
    questions: {
      team: { type: "choice", instructions: "Which team?", criteria: { billing: "Payments", technical: "Bugs", sales: null } },
      urgent: { type: "noul", instructions: "The message is urgent" },
      anger: { type: "score", instructions: "How angry?", criteria: ["Calm", "Annoyed", "Furious"] },
    },
  });
  assert.equal(response.model, "fake-1");
  assert.deepEqual(response.usage, { input_tokens: 30, output_tokens: 3 });

  const { team, urgent, anger } = response.answers;
  assert.ok(team?.type === "choice" && urgent?.type === "noul" && anger?.type === "score");
  assert.equal(team.choice, "technical");
  assert.deepEqual(team.probabilities, { billing: 0.05, technical: 0.9, sales: 0.05 });
  assert.equal(urgent.noul, 0.9); // fake falls back to the first label, "yes"
  assert.deepEqual(anger.legend, { "0": "Calm", "1": "Annoyed", "2": "Furious" });
  near(anger.score, 0.15);
  assert.ok(team.confidence > 0.5 && team.confidence < 1);
});

test("rotations present options in different orders and map results back", async () => {
  const backend = new FakeBackend("sales");
  const response = await systemOne(
    backend,
    { state: "s", questions: { q: { type: "choice", instructions: "?", criteria: { billing: null, technical: null, sales: null } } } },
    { ...DEFAULT_ENGINE_OPTIONS, rotations: 3 },
  );
  assert.equal(backend.prompts.length, 3);
  assert.equal(new Set(backend.prompts).size, 3);
  const q = response.answers.q;
  assert.ok(q?.type === "choice");
  assert.deepEqual(q.probabilities, { billing: 0.05, technical: 0.05, sales: 0.9 });
});

test("more options than labels: tournament finds the winner and respects concurrency", async () => {
  const criteria = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`option-${String(i).padStart(2, "0")}`, null]));
  const backend = new FakeBackend("option-37");
  const response = await systemOne(backend, { state: "s", questions: { q: { type: "choice", instructions: "?", criteria } } }, { ...DEFAULT_ENGINE_OPTIONS, concurrency: 2 });
  const q = response.answers.q;
  assert.ok(q?.type === "choice");
  assert.equal(q.choice, "option-37");
  assert.equal(backend.prompts.length, 4); // 3 chunks + 1 final
  assert.ok(backend.peak <= 2);
  near(Object.values(q.probabilities).reduce((a, b) => a + b, 0), 1, 0.01);
});

test("scored: a request is one batched call, plus one per extra tournament round, in any prompt order", async () => {
  const criteria = Object.fromEntries(Array.from({ length: 40 }, (_, i) => [`option-${String(i).padStart(2, "0")}`, null]));
  const request: SystemOneRequest = {
    state: "the state",
    questions: {
      big: { type: "choice", instructions: "?", criteria },
      flag: { type: "noul", instructions: "option-37 is mentioned" },
      level: { type: "score", instructions: "?", criteria: ["low", "option-37", "high"] },
    },
  };
  for (const order of PROMPT_ORDERS) {
    const backend = new FakeScoringBackend("option-37");
    const response = await systemOne(backend, request, { ...DEFAULT_ENGINE_OPTIONS, strategy: "scored", order });
    assert.deepEqual(backend.batches.map((b) => b.length), [5, 1]); // 3 chunks + noul + score, then the final round
    assert.deepEqual(untimed(response.gev), { strategy: "scored", model_calls: 2, repaired: 0 });
    assert.ok(response.gev!.model_ms <= response.gev!.ms);
    const { big, level } = response.answers;
    assert.ok(big?.type === "choice" && level?.type === "score");
    assert.equal(big.choice, "option-37");
    near(level.score, 1, 0.2);
  }
  await assert.rejects(systemOne(new FakeBackend("x"), request, { ...DEFAULT_ENGINE_OPTIONS, strategy: "scored" }), /cannot/);

  // Answered by name, the big choice is one prompt, plus a lettered one for each of the four sets of
  // ten names that share a first token, in the same moment as the rest: no second round.
  const wide = new FakeScoringBackend("option-37");
  const response = await systemOne(wide, request, { ...DEFAULT_ENGINE_OPTIONS, strategy: "scored", wideChoice: true });
  assert.deepEqual(wide.batches.map((b) => b.length).sort(), [1, 6]);
  assert.deepEqual(wide.tops.sort(), [4 + 40, undefined]);
  const big = response.answers.big;
  assert.ok(big?.type === "choice" && big.choice === "option-37");
  near(big.probabilities["option-37"]!, 0.9 * 0.9, 0.01);
});

test("prompt orders move the STATE later so more of the prompt is the same on every request", () => {
  const prompt = (order: (typeof PROMPT_ORDERS)[number]) => noulPrompt("the state", "it is raining", undefined, order);
  assert.ok(prompt("state-first").startsWith("STATE:\nthe state"));
  assert.ok(prompt("question-first").endsWith("STATE:\nthe state\n\nReply with exactly one of: yes, no"));
  assert.ok(prompt("state-last").endsWith("Reply with exactly one of: yes, no\n\nSTATE:\nthe state"));
  const frame = chatFrame("<bos><|turn>system\nGEVSYSTEMMARK <turn|>\n<|turn>user\nGEVUSERMARK<turn|>\n<|turn>model\n");
  assert.equal(framed(frame, " sys ", "user\n"), "<bos><|turn>system\nsys <turn|>\n<|turn>user\nuser<turn|>\n<|turn>model\n");
  assert.throws(() => chatFrame("no marks here"), /marks/);
});

test("packed: one model call answers the whole sheet; oversized choices go alone; skipped lines are repaired", async () => {
  const icons = Object.fromEntries(Array.from({ length: 20 }, (_, i) => [`icon-${String(i).padStart(2, "0")}`, null]));
  const request: SystemOneRequest = {
    state: "The API returns 500 on every call.",
    questions: {
      team: { type: "choice", instructions: "Which team?", criteria: { billing: "Payments", technical: "Bugs", sales: null } },
      urgent: { type: "noul", instructions: "The message is urgent" },
      anger: { type: "score", instructions: "How angry?", criteria: ["Calm", "Annoyed", "technical fury"] },
      icon: { type: "choice", instructions: "Which icon?", criteria: icons },
    },
  };
  const packed = { ...DEFAULT_ENGINE_OPTIONS, strategy: "packed" } as const;

  const backend = new FakeBackend("technical");
  const response = await systemOne(backend, request, packed);
  const isolated = await systemOne(new FakeBackend("technical"), request);
  assert.deepEqual(response.answers, isolated.answers);
  assert.deepEqual(Object.keys(response.answers), ["team", "urgent", "anger", "icon"]);
  // One sheet for three questions, plus the 20-icon tournament (2 chunks + 1 final).
  assert.deepEqual(untimed(response.gev), { strategy: "packed", model_calls: 4, repaired: 0 });
  assert.equal(isolated.gev?.model_calls, 6);

  const forgetful = new FakeBackend("technical");
  forgetful.skip.add(2);
  const repaired = await systemOne(forgetful, request, packed);
  assert.deepEqual(repaired.answers, isolated.answers);
  assert.deepEqual(untimed(repaired.gev), { strategy: "packed", model_calls: 5, repaired: 1 });
});

test("HTTP: auth, CORS, validation, and upstream error mapping", async () => {
  const post = (app: ReturnType<typeof createApp>, body: unknown, headers: Record<string, string> = {}) =>
    app.request("/v1/systemone", { method: "POST", headers: { "content-type": "application/json", ...headers }, body: JSON.stringify(body) });
  const valid = { state: "hi", model: "jev-latest", questions: { q: { type: "noul", instructions: "It is a greeting" } } };

  const app = createApp({ backend: new FakeBackend("x"), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set(["sk-test"]) });
  assert.equal((await post(app, valid)).status, 401);
  assert.equal((await post(app, valid, { authorization: "Bearer nope" })).status, 401);
  const ok = await post(app, valid, { authorization: "Bearer sk-test" });
  assert.equal(ok.status, 200);
  assert.deepEqual(((await ok.json()) as any).answers.q, { type: "noul", noul: 0.9 });

  const open = createApp({ backend: new FakeBackend("x"), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set() });
  for (const bad of [
    {},
    { state: "s", questions: {} },
    { state: "s", questions: { q: { type: "maybe", instructions: "?" } } },
    { state: "s", questions: { q: { type: "choice", instructions: "?", criteria: { only: null } } } },
    { state: "s", questions: { q: { type: "score", instructions: "?", criteria: ["one"] } } },
  ]) {
    const response = await post(open, bad);
    assert.equal(response.status, 422, JSON.stringify(bad));
    assert.equal(((await response.json()) as any).error.type, "invalid_request");
  }

  // CORS is wide open, and the preflight must pass without an API key.
  const preflight = await app.request("/v1/systemone", {
    method: "OPTIONS",
    headers: { origin: "http://localhost:5173", "access-control-request-method": "POST", "access-control-request-headers": "authorization,content-type" },
  });
  assert.equal(preflight.status, 204);
  assert.equal(preflight.headers.get("access-control-allow-origin"), "*");
  assert.match(preflight.headers.get("access-control-allow-headers") ?? "", /authorization/i);
  assert.equal(ok.headers.get("access-control-allow-origin"), "*");

  const failing = (status: number): Backend => ({
    model: "down",
    generate: async () => {
      throw new UpstreamError(status, "boom");
    },
  });
  const quiet = test.mock.method(console, "error", () => {});
  assert.equal((await post(createApp({ backend: failing(429), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set() }), valid)).status, 429);
  assert.equal((await post(createApp({ backend: failing(500), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set() }), valid)).status, 502);
  quiet.mock.restore();
});
