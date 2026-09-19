import type { Backend } from "./backends/backend.ts";
import { CHOICE_LABELS, NOUL_LABELS, type PromptOrder, SCORE_LABELS, SYSTEM_INSTRUCTION, choicePrompt, namedChoicePrompt, noulPrompt, scorePrompt } from "./prompt.ts";
import { type TokenLogprob, tokenKey, argmax, confidence, expectedLevel, labelDistribution, mean, normalize, round } from "./scoring.ts";
import { SHEET_SIZE, SHEET_SYSTEM_INSTRUCTION, type SheetQuestion, readSheet, sheetLabels, sheetMaxTokens, sheetPrompt } from "./sheet.ts";
import type { Answer, ChoiceQuestion, Json, Question, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage } from "./types.ts";

export type EngineOptions = {
  /** Max in-flight backend calls per request. */
  concurrency: number;
  /**
   * Letter labels carry position bias (models favor "A"). Asking each choice question
   * this many times with the options rotated, then averaging, trades calls for calibration.
   */
  rotations: number;
  /**
   * "isolated": one model call per question; answers cannot influence each other.
   * "packed": questions share one call and are answered as a sheet. Far fewer calls, but the
   * answers are produced jointly. Questions that don't fit a sheet, or that the model fails
   * to answer on it, fall back to isolated.
   * "scored": the isolated prompts, but every prompt that is ready at the same moment goes to the
   * model in one batched call that only reads the first answer token's distribution. Needs a
   * backend with `score` (an autoregressive model).
   */
  strategy: "isolated" | "packed" | "scored";
  /** Where the STATE goes in an isolated prompt; see PromptOrder. */
  order: PromptOrder;
  /**
   * With "scored": ask a choice that has more options than letters by name, in one prompt, and read
   * the first token of the name, instead of as a tournament, which costs a second model call.
   */
  wideChoice: boolean;
};

export const DEFAULT_ENGINE_OPTIONS: EngineOptions = { concurrency: 16, rotations: 1, strategy: "isolated", order: "state-first", wideChoice: false };

// How far past the number of names to read, so that a few off-script tokens don't push names out of view.
const WIDE_MARGIN = 40;

// Room for the label plus the end-of-turn token.
const ANSWER_TOKENS = 2;

type Option = [name: string, description: string | null];

/** A prompt waiting for a batched `score` call, and how many top logprobs it needs (default if undefined). */
type Pending = { prompt: string; top: number | undefined; resolve: (top: TokenLogprob[]) => void; reject: (error: unknown) => void };

class Run {
  readonly usage: Usage = { input_tokens: 0, output_tokens: 0 };
  modelCalls = 0;
  /** Sheet questions that had to be asked again on their own. */
  repaired = 0;
  /** Time with at least one model call in flight. */
  modelMs = 0;
  #inFlight = 0;
  #busySince = 0;
  readonly #backend: Backend;
  readonly #options: EngineOptions;
  readonly #state: Json;
  #active = 0;
  readonly #waiting: (() => void)[] = [];
  /** Prompts waiting for the next batched `score` call. */
  #batch: Pending[] = [];

  constructor(backend: Backend, options: EngineOptions, state: Json) {
    this.#backend = backend;
    this.#options = options;
    this.#state = state;
  }

  /** Overlapping calls count once, so this is wall time and not a sum over calls. */
  async #timed<T>(call: () => Promise<T>): Promise<T> {
    if (this.#inFlight++ === 0) this.#busySince = performance.now();
    try {
      return await call();
    } finally {
      if (--this.#inFlight === 0) this.modelMs += performance.now() - this.#busySince;
    }
  }

  async #generate(system: string, prompt: string, maxTokens: number) {
    // A finishing call hands its slot straight to the next waiter, so #active only moves when no one is queued.
    if (this.#active >= this.#options.concurrency) await new Promise<void>((resolve) => this.#waiting.push(resolve));
    else this.#active++;
    try {
      const { positions, usage } = await this.#timed(() => this.#backend.generate(system, prompt, maxTokens));
      this.modelCalls++;
      this.usage.input_tokens += usage.input_tokens;
      this.usage.output_tokens += usage.output_tokens;
      return positions;
    } finally {
      const next = this.#waiting.shift();
      if (next) next();
      else this.#active--;
    }
  }

  /**
   * Joins the batch that leaves once everything runnable right now has asked: all of a request's
   * questions the first time, a tournament's final round the second.
   */
  #score(prompt: string, top?: number): Promise<TokenLogprob[]> {
    return new Promise((resolve, reject) => {
      if (this.#batch.length === 0) setImmediate(() => this.#flush());
      this.#batch.push({ prompt, top, resolve, reject });
    });
  }

  /**
   * One call per logprobs width, sent together. A single call read as wide as its widest prompt was
   * measured far slower (plan suite: 308 ms against 176): vLLM's cost of wide logprobs is per prompt.
   */
  #flush() {
    const widths = new Set(this.#batch.map((b) => b.top));
    for (const top of widths) void this.#send(this.#batch.filter((b) => b.top === top), top);
    this.#batch = [];
  }

  async #send(batch: Pending[], top: number | undefined) {
    try {
      const { tops, usage } = await this.#timed(() => this.#backend.score!(SYSTEM_INSTRUCTION, batch.map((b) => b.prompt), top));
      this.modelCalls++;
      this.usage.input_tokens += usage.input_tokens;
      this.usage.output_tokens += usage.output_tokens;
      batch.forEach((b, i) => b.resolve(tops[i] ?? []));
    } catch (error) {
      batch.forEach((b) => b.reject(error));
    }
  }

  async #ask(prompt: string, labels: string[], top?: number): Promise<number[]> {
    if (this.#options.strategy === "scored") return labelDistribution(await this.#score(prompt, top), labels);
    const [first] = await this.#generate(SYSTEM_INSTRUCTION, prompt, ANSWER_TOKENS);
    return labelDistribution(first?.top ?? [], labels);
  }

  /** One model call for this question alone. */
  async isolated(question: Question): Promise<Answer> {
    switch (question.type) {
      case "noul":
        return toAnswer(question, await this.#ask(noulPrompt(this.#state, question.instructions, question.criteria, this.#options.order), NOUL_LABELS));
      case "score":
        return toAnswer(question, await this.#ask(scorePrompt(this.#state, question.instructions, question.criteria, this.#options.order), SCORE_LABELS.slice(0, question.criteria.length)));
      case "choice":
        return toAnswer(question, await this.#rank(question.instructions, Object.entries(question.criteria)));
    }
  }

  /** One model call for the whole sheet; returns the answers it could read. */
  async sheet(questions: SheetQuestion[]): Promise<Map<string, Answer>> {
    const positions = await this.#generate(SHEET_SYSTEM_INSTRUCTION, sheetPrompt(this.#state, questions), sheetMaxTokens(questions.length));
    const distributions = readSheet(positions, questions);
    return new Map(questions.flatMap((q) => (distributions.has(q.id) ? [[q.id, toAnswer(q.question, distributions.get(q.id)!)] as const] : [])));
  }

  /** Distribution over options that fit in one prompt, averaged over rotated orderings. */
  async #rankOnce(instructions: Json, options: Option[]): Promise<number[]> {
    const n = options.length;
    const rotations = Math.min(this.#options.rotations, n);
    const labels = CHOICE_LABELS.slice(0, n);
    const runs = await Promise.all(
      Array.from({ length: rotations }, async (_, r) => {
        const offset = Math.floor((r * n) / rotations);
        const rotated = options.map((_, i) => options[(i + offset) % n]!);
        const p = await this.#ask(choicePrompt(this.#state, instructions, rotated, this.#options.order), labels);
        return options.map((_, i) => p[(i - offset + n) % n]!);
      }),
    );
    return mean(runs);
  }

  /**
   * The model answers with the option's name and only the first token is read. Names that start
   * with the same token ("shopping_cart", "shopping_bag") share that token's probability, split by
   * a lettered question among just those names. Everything goes out in the same batch.
   */
  async #rankByName(instructions: Json, options: Option[]): Promise<number[]> {
    const firsts = (await this.#backend.firstTokens!(options.map(([name]) => name))).map(tokenKey);
    const groups = new Map<string, number[]>();
    firsts.forEach((first, i) => groups.set(first, [...(groups.get(first) ?? []), i]));
    const keys = [...groups.keys()];
    const [byGroup, ...within] = await Promise.all([
      this.#ask(namedChoicePrompt(this.#state, instructions, options, this.#options.order), keys, keys.length + WIDE_MARGIN),
      ...keys.map((key) => (groups.get(key)!.length > 1 ? this.#rank(instructions, groups.get(key)!.map((i) => options[i]!)) : [1])),
    ]);
    const probabilities = options.map(() => 0);
    keys.forEach((key, g) => groups.get(key)!.forEach((i, m) => (probabilities[i] = byGroup![g]! * within[g]![m]!)));
    return normalize(probabilities);
  }

  /**
   * More options than labels: rank each chunk, then rank the chunk winners against
   * each other. A non-winner keeps its within-chunk ratio to its chunk's winner.
   */
  async #rank(instructions: Json, options: Option[]): Promise<number[]> {
    const size = CHOICE_LABELS.length;
    if (options.length <= size) return this.#rankOnce(instructions, options);
    if (this.#options.wideChoice && this.#options.strategy === "scored" && this.#backend.firstTokens) return this.#rankByName(instructions, options);
    const chunks: Option[][] = [];
    for (let i = 0; i < options.length; i += size) chunks.push(options.slice(i, i + size));
    const within = await Promise.all(chunks.map((chunk) => (chunk.length > 1 ? this.#rankOnce(instructions, chunk) : [1])));
    const winners = within.map((p) => argmax(p));
    const final = await this.#rank(instructions, chunks.map((chunk, c) => chunk[winners[c]!]!));
    return normalize(within.flatMap((p, c) => p.map((pi) => (final[c]! * pi) / p[winners[c]!]!)));
  }
}

/** Builds the typed answer from a distribution over the question's options, in criteria order. */
function toAnswer(question: Question, probabilities: number[]): Answer {
  switch (question.type) {
    case "noul":
      return { type: "noul", noul: round(probabilities[0]!) };
    case "score":
      return scoreAnswer(question, probabilities);
    case "choice":
      return choiceAnswer(question, probabilities);
  }
}

function scoreAnswer(question: ScoreQuestion, probabilities: number[]): Answer {
  return {
    type: "score",
    score: round(expectedLevel(probabilities)),
    legend: Object.fromEntries(question.criteria.map((level, i) => [String(i), level])),
    probabilities: Object.fromEntries(probabilities.map((p, i) => [String(i), round(p)])),
    confidence: round(confidence(probabilities)),
  };
}

function choiceAnswer(question: ChoiceQuestion, probabilities: number[]): Answer {
  const names = Object.keys(question.criteria);
  return {
    type: "choice",
    choice: names[argmax(probabilities)]!,
    probabilities: Object.fromEntries(names.map((name, i) => [name, round(probabilities[i]!)])),
    confidence: round(confidence(probabilities)),
  };
}

async function answerPacked(run: Run, questions: [string, Question][]): Promise<[string, Answer][]> {
  const onSheet: SheetQuestion[] = [];
  const alone: [string, Question][] = [];
  for (const [id, question] of questions) {
    const labels = sheetLabels(question);
    if (labels) onSheet.push({ id, question, labels });
    else alone.push([id, question]);
  }
  // A sheet of one is just an isolated question with a worse prompt.
  if (onSheet.length < 2) alone.push(...onSheet.splice(0).map((q): [string, Question] => [q.id, q.question]));

  const sheets: SheetQuestion[][] = [];
  for (let i = 0; i < onSheet.length; i += SHEET_SIZE) sheets.push(onSheet.slice(i, i + SHEET_SIZE));

  const [read, rest] = await Promise.all([
    Promise.all(sheets.map((sheet) => run.sheet(sheet))),
    Promise.all(alone.map(async ([id, question]): Promise<[string, Answer]> => [id, await run.isolated(question)])),
  ]);
  const answers = new Map<string, Answer>([...read.flatMap((m) => [...m]), ...rest]);

  // Anything the model skipped or answered off-format is asked again on its own.
  const unread = onSheet.filter((q) => !answers.has(q.id));
  run.repaired = unread.length;
  for (const [id, answer] of await Promise.all(unread.map(async (q): Promise<[string, Answer]> => [q.id, await run.isolated(q.question)]))) answers.set(id, answer);

  return questions.map(([id]) => [id, answers.get(id)!]);
}

export async function systemOne(backend: Backend, request: SystemOneRequest, options: EngineOptions = DEFAULT_ENGINE_OPTIONS): Promise<SystemOneResponse> {
  if (options.strategy === "scored" && !backend.score) throw new Error(`The "scored" strategy needs a backend that can score prompts; ${backend.model} cannot`);
  const started = performance.now();
  const run = new Run(backend, options, request.state);
  const questions = Object.entries(request.questions);
  const entries =
    options.strategy === "packed"
      ? await answerPacked(run, questions)
      : await Promise.all(questions.map(async ([id, question]): Promise<[string, Answer]> => [id, await run.isolated(question)]));
  return {
    model: backend.model,
    answers: Object.fromEntries(entries),
    usage: run.usage,
    gev: { strategy: options.strategy, model_calls: run.modelCalls, repaired: run.repaired, ms: Math.round(performance.now() - started), model_ms: Math.round(run.modelMs) },
  };
}
