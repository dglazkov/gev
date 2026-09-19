import assert from "node:assert/strict";
import { test } from "node:test";
import { createApp } from "./app.ts";
import { type Backend, type FirstTokenResult, UpstreamError } from "./backends/backend.ts";
import { DEFAULT_ENGINE_OPTIONS, systemOne } from "./engine.ts";
import { confidence, expectedLevel, labelDistribution } from "./scoring.ts";

const near = (actual: number, expected: number, eps = 1e-3) =>
  assert.ok(Math.abs(actual - expected) < eps, `expected ${actual} ≈ ${expected}`);

/** Answers each prompt by finding which label sits next to `favorite` and putting `p` of the mass there. */
class FakeBackend implements Backend {
  readonly model = "fake-1";
  readonly prompts: string[] = [];
  readonly #favorite: string;
  readonly #p: number;
  inflight = 0;
  peak = 0;

  constructor(favorite: string, p = 0.9) {
    this.#favorite = favorite;
    this.#p = p;
  }

  async firstToken(_system: string, prompt: string): Promise<FirstTokenResult> {
    this.prompts.push(prompt);
    this.peak = Math.max(this.peak, ++this.inflight);
    await new Promise((resolve) => setImmediate(resolve));
    this.inflight--;
    const labels = prompt.split("\n").at(-1)!.replace("Reply with exactly one of: ", "").split(", ");
    const line = prompt.split("\n").find((l) => /^\w+: /.test(l) && l.includes(this.#favorite));
    const favorite = line?.split(":")[0] ?? labels[0]!;
    const rest = (1 - this.#p) / (labels.length - 1);
    return {
      top: labels.map((label) => ({ token: label, logprob: Math.log(label === favorite ? this.#p : rest) })),
      usage: { input_tokens: 10, output_tokens: 1 },
    };
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
  const response = await systemOne(backend, { state: "s", questions: { q: { type: "choice", instructions: "?", criteria } } }, { concurrency: 2, rotations: 1 });
  const q = response.answers.q;
  assert.ok(q?.type === "choice");
  assert.equal(q.choice, "option-37");
  assert.equal(backend.prompts.length, 4); // 3 chunks + 1 final
  assert.ok(backend.peak <= 2);
  near(Object.values(q.probabilities).reduce((a, b) => a + b, 0), 1, 0.01);
});

test("HTTP: auth, validation, and upstream error mapping", async () => {
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

  const failing = (status: number): Backend => ({
    model: "down",
    firstToken: async () => {
      throw new UpstreamError(status, "boom");
    },
  });
  const quiet = test.mock.method(console, "error", () => {});
  assert.equal((await post(createApp({ backend: failing(429), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set() }), valid)).status, 429);
  assert.equal((await post(createApp({ backend: failing(500), engine: DEFAULT_ENGINE_OPTIONS, apiKeys: new Set() }), valid)).status, 502);
  quiet.mock.restore();
});
