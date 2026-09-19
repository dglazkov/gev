import { GoogleAuth } from "google-auth-library";
import { type Backend, type Generation, type Position, type Scores, TOP_LOGPROBS, postJson } from "./backend.ts";

export type VllmOptions = {
  /** e.g. https://gev-model-xyz.a.run.app/v1 */
  baseUrl: string;
  model: string;
  apiKey?: string;
  /** Authenticate with a Google ID token, for a private Cloud Run service. */
  gcpIdToken?: boolean;
};

type WirePosition = { token: string; top_logprobs: { token: string; logprob: number }[] };

// Even with thinking disabled, Gemma 4 models open with an empty thought channel before the answer:
// "<|channel>thought\n<channel|>". DiffusionGemma leaves off the closing marker about a third of
// the time and goes straight into the answer, so the preamble is matched token by token rather
// than as a delimited block.
const PREAMBLE_TOKENS = 6;
const PREAMBLE = new Set(["<|channel>", "<channel|>", "thought"]);

/** Drops the empty thought preamble, leaving only the positions of the answer itself. */
export function answerPositions(positions: WirePosition[]): Position[] {
  let start = 0;
  while (start < positions.length && (PREAMBLE.has(positions[start]!.token) || positions[start]!.token.trim() === "")) start++;
  return positions.slice(start).map((position) => ({
    token: position.token,
    top: position.top_logprobs.map((c) => ({ token: c.token, logprob: c.logprob })),
  }));
}

// Stand-ins for the two messages when asking the server what its chat template renders around them.
const SYSTEM_MARK = "GEVSYSTEMMARK";
const USER_MARK = "GEVUSERMARK";

/** The text a chat template puts before the system message, between it and the user message, and after. */
export type ChatFrame = [head: string, middle: string, tail: string];

/** Splits a rendering of the two marks into the frame around them. */
export function chatFrame(rendered: string): ChatFrame {
  const [head, rest] = rendered.split(SYSTEM_MARK);
  const [middle, tail] = (rest ?? "").split(USER_MARK);
  if (head === undefined || middle === undefined || tail === undefined) throw new Error(`Could not find the marks in the rendered chat template: ${rendered.slice(0, 200)}`);
  return [head, middle, tail];
}

/** The template trims both messages, so the frame must meet trimmed text to render the same tokens. */
export const framed = ([head, middle, tail]: ChatFrame, system: string, prompt: string) => `${head}${system.trim()}${middle}${prompt.trim()}${tail}`;

/** Gemma served by vLLM's OpenAI-compatible chat completions API (or anything that speaks it and returns top_logprobs). */
export class VllmBackend implements Backend {
  readonly model: string;
  readonly #options: VllmOptions;
  readonly #auth = new GoogleAuth();
  #frame: Promise<ChatFrame> | undefined;
  #idTokenClient: ReturnType<GoogleAuth["getIdTokenClient"]> | undefined;

  constructor(options: VllmOptions) {
    this.#options = { ...options, baseUrl: options.baseUrl.replace(/\/+$/, "") };
    this.model = options.model;
  }

  async #headers(): Promise<Record<string, string>> {
    if (this.#options.gcpIdToken) {
      // One client for the life of the process: it keeps its token until shortly before expiry,
      // where a new client would go back to the metadata server on every model call.
      this.#idTokenClient ??= this.#auth.getIdTokenClient(new URL(this.#options.baseUrl).origin);
      return Object.fromEntries(await (await this.#idTokenClient).getRequestHeaders());
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

  /**
   * The server's own rendering of its chat template, asked for once. Gemma 4 templates differ
   * between models and revisions (a trailing space after the system text, whether the empty thought
   * channel is part of the generation prompt), and a prompt that is off by one token answers "The".
   */
  #chatFrame(): Promise<ChatFrame> {
    this.#frame ??= (async () => {
      const root = this.#options.baseUrl.replace(/\/v1$/, "");
      const { tokens } = await postJson(`${root}/tokenize`, await this.#headers(), {
        model: this.#options.model,
        messages: [
          { role: "system", content: SYSTEM_MARK },
          { role: "user", content: USER_MARK },
        ],
        add_generation_prompt: true,
        chat_template_kwargs: { enable_thinking: false },
      });
      const { prompt } = await postJson(`${root}/detokenize`, await this.#headers(), { model: this.#options.model, tokens });
      return chatFrame(prompt);
    })();
    // A failure (say, a model server that is still starting) must not be remembered.
    this.#frame.catch(() => (this.#frame = undefined));
    return this.#frame;
  }

  async score(system: string, prompts: string[]): Promise<Scores> {
    const frame = await this.#chatFrame();
    // The raw completions endpoint takes all the prompts in one request and schedules them as one
    // batch. Chat completions would need a request per prompt and re-render the template each time.
    const data = await postJson(`${this.#options.baseUrl}/completions`, await this.#headers(), {
      model: this.#options.model,
      prompt: prompts.map((prompt) => framed(frame, system, prompt)),
      // The frame already starts with <bos>.
      add_special_tokens: false,
      temperature: 0,
      max_tokens: 1,
      logprobs: TOP_LOGPROBS,
    });
    const tops: Scores["tops"] = prompts.map(() => []);
    for (const choice of data.choices ?? []) {
      const top: Record<string, number> = choice.logprobs?.top_logprobs?.[0] ?? {};
      tops[choice.index] = Object.entries(top).map(([token, logprob]) => ({ token, logprob }));
    }
    return { tops, usage: { input_tokens: data.usage?.prompt_tokens ?? 0, output_tokens: data.usage?.completion_tokens ?? 0 } };
  }
}
