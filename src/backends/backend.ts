import type { TokenLogprob } from "../scoring.ts";
import type { Usage } from "../types.ts";

export const TOP_LOGPROBS = 20;

/** One generated token, with the top-k candidates the model weighed at that position. */
export type Position = { token: string; top: TokenLogprob[] };

export type Generation = { positions: Position[]; usage: Usage };

/** `tops[i]` belongs to `prompts[i]`. */
export type Scores = { tops: TokenLogprob[][]; usage: Usage };

/** A language model that reports top-k logprobs for every token of its answer. */
export interface Backend {
  /** Reported in the response `model` field. */
  readonly model: string;
  /** Greedy generation of up to `maxTokens` answer tokens (any reasoning preamble excluded). */
  generate(system: string, prompt: string, maxTokens: number): Promise<Generation>;
  /**
   * The top-k candidates for the first answer token of every prompt, in one batched model call
   * and without generating anything further. Only an autoregressive model can do this: one forward
   * pass over a prompt already yields the distribution of the token that follows it.
   */
  score?(system: string, prompts: string[]): Promise<Scores>;
}

export class UpstreamError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "UpstreamError";
    this.status = status;
  }
}

const RETRYABLE = new Set([429, 500, 502, 503, 504]);

/** POST JSON with exponential backoff on rate limits and transient upstream failures. */
export async function postJson(url: string, headers: Record<string, string>, body: unknown, retries = 3): Promise<any> {
  for (let attempt = 0; ; attempt++) {
    const response = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json", ...headers },
      body: JSON.stringify(body),
    });
    if (response.ok) return response.json();
    if (!RETRYABLE.has(response.status) || attempt >= retries) {
      throw new UpstreamError(response.status, `${response.status} from ${new URL(url).host}: ${(await response.text()).slice(0, 500)}`);
    }
    await new Promise((resolve) => setTimeout(resolve, 250 * 2 ** attempt * (0.5 + Math.random())));
  }
}
