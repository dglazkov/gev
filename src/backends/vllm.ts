import { GoogleAuth } from "google-auth-library";
import { type Backend, type Generation, type Position, TOP_LOGPROBS, postJson } from "./backend.ts";

export type VllmOptions = {
  /** e.g. https://gev-model-xyz.a.run.app/v1 */
  baseUrl: string;
  model: string;
  apiKey?: string;
  /** Authenticate with a Google ID token, for a private Cloud Run service. */
  gcpIdToken?: boolean;
};

type WirePosition = { token: string; top_logprobs: { token: string; logprob: number }[] };

// Even with thinking disabled, Gemma 4 models open with an empty thought channel
// ("<|channel>thought\n<channel|>") before the answer, so the answer is not the first token.
const PREAMBLE_TOKENS = 6;

/** Drops <|channel>…<channel|> blocks, leaving only the positions of the answer itself. */
export function answerPositions(positions: WirePosition[]): Position[] {
  const answer: Position[] = [];
  let inChannel = false;
  for (const position of positions) {
    if (position.token === "<|channel>") inChannel = true;
    else if (position.token === "<channel|>") inChannel = false;
    else if (!inChannel) answer.push({ token: position.token, top: position.top_logprobs.map((c) => ({ token: c.token, logprob: c.logprob })) });
  }
  return answer;
}

/** Gemma served by vLLM's OpenAI-compatible chat completions API (or anything that speaks it and returns top_logprobs). */
export class VllmBackend implements Backend {
  readonly model: string;
  readonly #options: VllmOptions;
  readonly #auth = new GoogleAuth();

  constructor(options: VllmOptions) {
    this.#options = { ...options, baseUrl: options.baseUrl.replace(/\/+$/, "") };
    this.model = options.model;
  }

  async #headers(): Promise<Record<string, string>> {
    if (this.#options.gcpIdToken) {
      const client = await this.#auth.getIdTokenClient(new URL(this.#options.baseUrl).origin);
      return Object.fromEntries(await client.getRequestHeaders());
    }
    return this.#options.apiKey ? { authorization: `Bearer ${this.#options.apiKey}` } : {};
  }

  async generate(system: string, prompt: string, maxTokens: number): Promise<Generation> {
    const data = await postJson(`${this.#options.baseUrl}/chat/completions`, await this.#headers(), {
      model: this.#options.model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      temperature: 0,
      max_tokens: maxTokens + PREAMBLE_TOKENS,
      logprobs: true,
      top_logprobs: TOP_LOGPROBS,
      chat_template_kwargs: { enable_thinking: false },
    });
    return {
      positions: answerPositions(data.choices?.[0]?.logprobs?.content ?? []),
      usage: {
        input_tokens: data.usage?.prompt_tokens ?? 0,
        output_tokens: data.usage?.completion_tokens ?? 0,
      },
    };
  }
}
