import type { Backend } from "./backends/backend.ts";
import { VllmBackend } from "./backends/vllm.ts";
import { DEFAULT_ENGINE_OPTIONS, type EngineOptions } from "./engine.ts";
import { PROMPT_ORDERS, type PromptOrder } from "./prompt.ts";

type Env = Record<string, string | undefined>;

export const DEFAULT_MODEL = "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic";

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

/**
 * GEV_TEMPERATURE: one number for every question type ("2.5"), or per type ("choice=3,score=2,noul=2.5").
 * Parts may also be separated by ";", because gcloud splits --set-env-vars values on commas.
 */
function temperatureFromEnv(env: Env): EngineOptions["temperature"] {
  const temperature = { ...DEFAULT_ENGINE_OPTIONS.temperature };
  for (const part of (env.GEV_TEMPERATURE ?? "").split(/[,;]/).filter(Boolean)) {
    const [type, raw] = part.includes("=") ? part.split("=") : [undefined, part];
    const value = Number(raw);
    if (!(value > 0) || (type !== undefined && !(type in temperature))) throw new Error(`GEV_TEMPERATURE must be a positive number or choice=…,score=…,noul=…, got "${env.GEV_TEMPERATURE}"`);
    for (const key of Object.keys(temperature) as (keyof typeof temperature)[]) if (type === undefined || type === key) temperature[key] = value;
  }
  return temperature;
}

export function engineOptionsFromEnv(env: Env = process.env): EngineOptions {
  const strategy = env.GEV_STRATEGY ?? DEFAULT_ENGINE_OPTIONS.strategy;
  if (strategy !== "isolated" && strategy !== "packed" && strategy !== "scored") throw new Error(`GEV_STRATEGY must be "isolated", "packed" or "scored", got "${strategy}"`);
  const order = (env.GEV_PROMPT_ORDER ?? DEFAULT_ENGINE_OPTIONS.order) as PromptOrder;
  if (!PROMPT_ORDERS.includes(order)) throw new Error(`GEV_PROMPT_ORDER must be one of ${PROMPT_ORDERS.join(", ")}, got "${order}"`);
  return {
    strategy,
    order,
    wideChoice: env.GEV_WIDE_CHOICE === "1",
    temperature: temperatureFromEnv(env),
    concurrency: positiveInt(env, "GEV_CONCURRENCY", DEFAULT_ENGINE_OPTIONS.concurrency),
    rotations: positiveInt(env, "GEV_ROTATIONS", DEFAULT_ENGINE_OPTIONS.rotations),
  };
}

/** Comma-separated bearer tokens. Empty means the API is open (rely on Cloud Run IAM instead). */
export function apiKeysFromEnv(env: Env = process.env): Set<string> {
  return new Set((env.GEV_API_KEYS ?? "").split(",").map((k) => k.trim()).filter(Boolean));
}
