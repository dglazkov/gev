import type { TokenLogprob } from "../scoring.ts";
import type { Usage } from "../types.ts";

export const TOP_LOGPROBS = 20;

export type FirstTokenResult = { top: TokenLogprob[]; usage: Usage };

/** A language model that can report the top-k logprobs of the first token it would generate. */
export interface Backend {
  /** Reported in the response `model` field. */
  readonly model: string;
  firstToken(system: string, prompt: string): Promise<FirstTokenResult>;
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
