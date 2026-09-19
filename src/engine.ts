import type { Backend } from "./backends/backend.ts";
import { CHOICE_LABELS, NOUL_LABELS, SCORE_LABELS, SYSTEM_INSTRUCTION, choicePrompt, noulPrompt, scorePrompt } from "./prompt.ts";
import { argmax, confidence, expectedLevel, labelDistribution, mean, normalize, round } from "./scoring.ts";
import type { Answer, ChoiceQuestion, Json, NoulQuestion, Question, ScoreQuestion, SystemOneRequest, SystemOneResponse, Usage } from "./types.ts";

export type EngineOptions = {
  /** Max in-flight backend calls per request. */
  concurrency: number;
  /**
   * Letter labels carry position bias (models favor "A"). Asking each choice question
   * this many times with the options rotated, then averaging, trades calls for calibration.
   */
  rotations: number;
};

export const DEFAULT_ENGINE_OPTIONS: EngineOptions = { concurrency: 16, rotations: 1 };

type Option = [name: string, description: string | null];

class Run {
  readonly usage: Usage = { input_tokens: 0, output_tokens: 0 };
  readonly #backend: Backend;
  readonly #options: EngineOptions;
  readonly #state: Json;
  #active = 0;
  readonly #waiting: (() => void)[] = [];

  constructor(backend: Backend, options: EngineOptions, state: Json) {
    this.#backend = backend;
    this.#options = options;
    this.#state = state;
  }

  async #ask(prompt: string, labels: string[]): Promise<number[]> {
    // A finishing call hands its slot straight to the next waiter, so #active only moves when no one is queued.
    if (this.#active >= this.#options.concurrency) await new Promise<void>((resolve) => this.#waiting.push(resolve));
    else this.#active++;
    try {
      const { top, usage } = await this.#backend.firstToken(SYSTEM_INSTRUCTION, prompt);
      this.usage.input_tokens += usage.input_tokens;
      this.usage.output_tokens += usage.output_tokens;
      return labelDistribution(top, labels);
    } finally {
      const next = this.#waiting.shift();
      if (next) next();
      else this.#active--;
    }
  }

  answer(question: Question): Promise<Answer> {
    switch (question.type) {
      case "noul":
        return this.#noul(question);
      case "choice":
        return this.#choice(question);
      case "score":
        return this.#score(question);
    }
  }

  async #noul(question: NoulQuestion): Promise<Answer> {
    const [yes] = await this.#ask(noulPrompt(this.#state, question.instructions, question.criteria), NOUL_LABELS);
    return { type: "noul", noul: round(yes!) };
  }

  async #score(question: ScoreQuestion): Promise<Answer> {
    const levels = question.criteria;
    const probabilities = await this.#ask(scorePrompt(this.#state, question.instructions, levels), SCORE_LABELS.slice(0, levels.length));
    return {
      type: "score",
      score: round(expectedLevel(probabilities)),
      legend: Object.fromEntries(levels.map((level, i) => [String(i), level])),
      probabilities: Object.fromEntries(probabilities.map((p, i) => [String(i), round(p)])),
      confidence: round(confidence(probabilities)),
    };
  }

  async #choice(question: ChoiceQuestion): Promise<Answer> {
    const options = Object.entries(question.criteria);
    const probabilities = await this.#rank(question.instructions, options);
    return {
      type: "choice",
      choice: options[argmax(probabilities)]![0],
      probabilities: Object.fromEntries(options.map(([name], i) => [name, round(probabilities[i]!)])),
      confidence: round(confidence(probabilities)),
    };
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
        const p = await this.#ask(choicePrompt(this.#state, instructions, rotated), labels);
        return options.map((_, i) => p[(i - offset + n) % n]!);
      }),
    );
    return mean(runs);
  }

  /**
   * More options than labels: rank each chunk, then rank the chunk winners against
   * each other. A non-winner keeps its within-chunk ratio to its chunk's winner.
   */
  async #rank(instructions: Json, options: Option[]): Promise<number[]> {
    const size = CHOICE_LABELS.length;
    if (options.length <= size) return this.#rankOnce(instructions, options);
    const chunks: Option[][] = [];
    for (let i = 0; i < options.length; i += size) chunks.push(options.slice(i, i + size));
    const within = await Promise.all(chunks.map((chunk) => (chunk.length > 1 ? this.#rankOnce(instructions, chunk) : [1])));
    const winners = within.map((p) => argmax(p));
    const final = await this.#rank(instructions, chunks.map((chunk, c) => chunk[winners[c]!]!));
    return normalize(within.flatMap((p, c) => p.map((pi) => (final[c]! * pi) / p[winners[c]!]!)));
  }
}

export async function systemOne(backend: Backend, request: SystemOneRequest, options: EngineOptions = DEFAULT_ENGINE_OPTIONS): Promise<SystemOneResponse> {
  const run = new Run(backend, options, request.state);
  const entries = await Promise.all(
    Object.entries(request.questions).map(async ([id, question]) => [id, await run.answer(question)] as const),
  );
  return { model: backend.model, answers: Object.fromEntries(entries), usage: run.usage };
}
