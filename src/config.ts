import type { Backend } from "./backends/backend.ts";
import { VllmBackend } from "./backends/vllm.ts";
import { DEFAULT_ENGINE_OPTIONS, type EngineOptions } from "./engine.ts";
import { PROMPT_ORDERS, type PromptOrder } from "./prompt.ts";

type Env = Record<string, string | undefined>;

export const DEFAULT_MODEL = "google/diffusiongemma-26B-A4B-it";

function positiveInt(env: Env, name: string, fallback: number): number {
  const raw = env[name];
  if (!raw) return fallback;
  const value = Number(raw);
  if (!Number.isInteger(value) || value < 1) throw new Error(`${name} must be a positive integer, got "${raw}"`);
  return value;
}

export function backendFromEnv(env: Env = process.env): Backend {
  if (!env.GEV_MODEL_URL) throw new Error("GEV_MODEL_URL is required: the model server's OpenAI-compatible base URL, e.g. https://gev-model-….run.app/v1");
  return new VllmBackend({
    baseUrl: env.GEV_MODEL_URL,
    model: env.GEV_MODEL ?? DEFAULT_MODEL,
    apiKey: env.GEV_MODEL_API_KEY,
    gcpIdToken: env.GEV_MODEL_GCP_AUTH === "1",
  });
}

export function engineOptionsFromEnv(env: Env = process.env): EngineOptions {
  const strategy = env.GEV_STRATEGY ?? DEFAULT_ENGINE_OPTIONS.strategy;
  if (strategy !== "isolated" && strategy !== "packed" && strategy !== "scored") throw new Error(`GEV_STRATEGY must be "isolated", "packed" or "scored", got "${strategy}"`);
  const order = (env.GEV_PROMPT_ORDER ?? DEFAULT_ENGINE_OPTIONS.order) as PromptOrder;
  if (!PROMPT_ORDERS.includes(order)) throw new Error(`GEV_PROMPT_ORDER must be one of ${PROMPT_ORDERS.join(", ")}, got "${order}"`);
  return {
    strategy,
    order,
    concurrency: positiveInt(env, "GEV_CONCURRENCY", DEFAULT_ENGINE_OPTIONS.concurrency),
    rotations: positiveInt(env, "GEV_ROTATIONS", DEFAULT_ENGINE_OPTIONS.rotations),
  };
}

/** Comma-separated bearer tokens. Empty means the API is open (rely on Cloud Run IAM instead). */
export function apiKeysFromEnv(env: Env = process.env): Set<string> {
  return new Set((env.GEV_API_KEYS ?? "").split(",").map((k) => k.trim()).filter(Boolean));
}
