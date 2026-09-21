import { GoogleAuth } from "google-auth-library";
import { type Backend, type Generation, type Position, type Scores, TOP_LOGPROBS, postJson } from "./backend.ts";

export const MODEL_SERVERS = ["vllm", "sglang"] as const;
export type ModelServer = (typeof MODEL_SERVERS)[number];

export type VllmOptions = {
  /** e.g. https://gev-model-xyz.a.run.app/v1 */
  baseUrl: string;
  model: string;
  apiKey?: string;
  /** Authenticate with a Google ID token, for a private Cloud Run service. */
  gcpIdToken?: boolean;
  /** SGLang speaks the same API but differs in its tokenizer endpoints and completions. Default vllm. */
  server?: ModelServer;
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

/** Gemma served by vLLM's OpenAI-compatible API, or by SGLang's (`server: "sglang"`). */
export class VllmBackend implements Backend {
  readonly model: string;
  readonly #options: VllmOptions;
  /** The tokenizer endpoints live at the server root, not under /v1. */
  readonly #root: string;
  readonly #sglang: boolean;
  readonly #auth = new GoogleAuth();
  #frame: Promise<ChatFrame> | undefined;
  #idTokenClient: ReturnType<GoogleAuth["getIdTokenClient"]> | undefined;
  readonly #firstTokens = new Map<string, Promise<string[]>>();

  constructor(options: VllmOptions) {
    this.#options = { ...options, baseUrl: options.baseUrl.replace(/\/+$/, "") };
    this.#root = this.#options.baseUrl.replace(/\/v1$/, "");
    this.#sglang = options.server === "sglang";
    this.model = options.model;
  }

  async #post(path: string, body: object): Promise<any> {
    return postJson(`${this.#root}${path}`, await this.#headers(), { model: this.#options.model, ...body });
  }

  /** vLLM answers with `prompt`. SGLang answers with `text`, and drops special tokens unless told not to. */
  async #detokenize(tokens: number[]): Promise<string> {
    if (!this.#sglang) return (await this.#post("/detokenize", { tokens })).prompt;
    return (await this.#post("/detokenize", { tokens, skip_special_tokens: false })).text;
  }

  /** Each token's text. SGLang's /tokenize can't return it, so each token is decoded on its own. */
  async #tokenStrings(text: string): Promise<string[]> {
    if (!this.#sglang) return (await this.#post("/tokenize", { prompt: text, add_special_tokens: false, return_token_strs: true })).token_strs;
    const { tokens } = await this.#post("/tokenize", { prompt: text, add_special_tokens: false });
    return (await this.#post("/detokenize", { tokens: (tokens as number[]).map((t) => [t]), skip_special_tokens: false })).text;
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
      const { tokens } = await this.#post("/tokenize", {
        messages: [
          { role: "system", content: SYSTEM_MARK },
          { role: "user", content: USER_MARK },
        ],
        add_generation_prompt: true,
        chat_template_kwargs: { enable_thinking: false },
      });
      const rendered = await this.#detokenize(tokens);
      if (this.#sglang) {
        // SGLang's completions can't be told add_special_tokens: false. Its Gemma 4 tokenizer adds
        // nothing (the frame's own <bos> is the only one), but a tokenizer that did would put a
        // second <bos> in front of every prompt and answer everything a little differently.
        const again = (await this.#post("/tokenize", { prompt: rendered, add_special_tokens: true })).tokens;
        if (again.length !== tokens.length) throw new Error(`SGLang tokenizes the chat frame into ${again.length} tokens, not ${tokens.length}: its tokenizer adds special tokens of its own`);
      }
      return chatFrame(rendered);
    })();
    // A failure (say, a model server that is still starting) must not be remembered.
    this.#frame.catch(() => (this.#frame = undefined));
    return this.#frame;
  }

  /** One /tokenize call for the whole list, remembered: an app asks about the same options every time. */
  firstTokens(strings: string[]): Promise<string[]> {
    const key = strings.join("\n");
    let found = this.#firstTokens.get(key);
    if (!found) {
      found = (async () => {
        // The first token of each line. A newline can share a token with what precedes it, never with what follows.
        const firsts: string[] = [];
        let atStart = true;
        for (const token of await this.#tokenStrings(key)) {
          if (atStart && token.trim() !== "") firsts.push(token);
          atStart = token.endsWith("\n");
        }
        if (firsts.length !== strings.length) throw new Error(`Expected ${strings.length} first tokens, found ${firsts.length}`);
        return firsts;
      })();
      this.#firstTokens.set(key, found);
      found.catch(() => this.#firstTokens.delete(key));
    }
    return found;
  }

  async score(system: string, prompts: string[], top = TOP_LOGPROBS): Promise<Scores> {
    const frame = await this.#chatFrame();
    // The raw completions endpoint takes all the prompts in one request and schedules them as one
    // batch. Chat completions would need a request per prompt and re-render the template each time.
    const data = await postJson(`${this.#options.baseUrl}/completions`, await this.#headers(), {
      model: this.#options.model,
      prompt: prompts.map((prompt) => framed(frame, system, prompt)),
      // The frame already starts with <bos>. SGLang has no such flag and adds nothing (see #chatFrame).
      ...(this.#sglang ? {} : { add_special_tokens: false }),
      temperature: 0,
      max_tokens: 1,
      logprobs: top,
    });
    const tops: Scores["tops"] = prompts.map(() => []);
    for (const choice of data.choices ?? []) {
      const top: Record<string, number> = choice.logprobs?.top_logprobs?.[0] ?? {};
      tops[choice.index] = Object.entries(top).map(([token, logprob]) => ({ token, logprob }));
    }
    return { tops, usage: { input_tokens: data.usage?.prompt_tokens ?? 0, output_tokens: data.usage?.completion_tokens ?? 0 } };
  }
}
