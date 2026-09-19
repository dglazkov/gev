import { GoogleAuth } from "google-auth-library";
import { type Backend, type FirstTokenResult, TOP_LOGPROBS, postJson } from "./backend.ts";

export type VllmOptions = {
  /** e.g. https://gev-model-xyz.a.run.app/v1 */
  baseUrl: string;
  model: string;
  apiKey?: string;
  /** Authenticate with a Google ID token, for a private Cloud Run service. */
  gcpIdToken?: boolean;
};

type Position = { token: string; top_logprobs: { token: string; logprob: number }[] };

// Even with thinking disabled, Gemma 4 models open with an empty thought channel
// ("<|channel>thought\n<channel|>") before the answer, so the label is not the first token.
const ANSWER_WINDOW = 8;

/** The first generated position outside a <|channel>…<channel|> block: where the answer label is. */
export function answerPosition(positions: Position[]): Position | undefined {
  let inChannel = false;
  for (const position of positions) {
    if (position.token === "<|channel>") inChannel = true;
    else if (position.token === "<channel|>") inChannel = false;
    else if (!inChannel) return position;
  }
  return undefined;
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

  async firstToken(system: string, prompt: string): Promise<FirstTokenResult> {
    const data = await postJson(`${this.#options.baseUrl}/chat/completions`, await this.#headers(), {
      model: this.#options.model,
      messages: [
        { role: "system", content: system },
        { role: "user", content: prompt },
      ],
      temperature: 0,
      max_tokens: ANSWER_WINDOW,
      logprobs: true,
      top_logprobs: TOP_LOGPROBS,
      chat_template_kwargs: { enable_thinking: false },
    });
    const top = answerPosition(data.choices?.[0]?.logprobs?.content ?? [])?.top_logprobs ?? [];
    return {
      top: top.map((c: { token: string; logprob: number }) => ({ token: c.token, logprob: c.logprob })),
      usage: {
        input_tokens: data.usage?.prompt_tokens ?? 0,
        output_tokens: data.usage?.completion_tokens ?? 0,
      },
    };
  }
}
