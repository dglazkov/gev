import { readFile } from "node:fs/promises";
import { Hono } from "hono";
import { cors } from "hono/cors";
import { type Backend, UpstreamError } from "./backends/backend.ts";
import { type EngineOptions, systemOne } from "./engine.ts";
import { parseRequest } from "./schema.ts";

export type AppOptions = { backend: Backend; engine: EngineOptions; apiKeys: Set<string> };

const error = (type: string, message: string, extra: object = {}) => ({ error: { type, message, ...extra } });

export function createApp({ backend, engine, apiKeys }: AppOptions): Hono {
  const app = new Hono();

  // Wide open: browser frontends on any origin may call the API. The bearer key is the only gate.
  app.use("/v1/*", cors({ origin: "*", allowHeaders: ["authorization", "content-type"], maxAge: 86400 }));

  app.get("/healthz", (c) => c.json({ ok: true, model: backend.model }));

  app.get("/", async (c) => c.html(await readFile(new URL("../public/index.html", import.meta.url), "utf8")));

  app.post("/v1/systemone", async (c) => {
    if (apiKeys.size > 0) {
      const key = c.req.header("authorization")?.match(/^Bearer\s+(.+)$/i)?.[1];
      if (!key || !apiKeys.has(key)) return c.json(error("unauthorized", "Missing or invalid API key"), 401);
    }

    let body: unknown;
    try {
      body = await c.req.json();
    } catch {
      return c.json(error("invalid_request", "Request body must be valid JSON"), 422);
    }
    const parsed = parseRequest(body);
    if (!parsed.ok) return c.json(error("invalid_request", "Request validation failed", { issues: parsed.issues }), 422);

    try {
      return c.json(await systemOne(backend, parsed.value, engine));
    } catch (e) {
      if (e instanceof UpstreamError && e.status === 429) return c.json(error("rate_limited", "Model backend rate limit exceeded"), 429);
      console.error(JSON.stringify({ severity: "ERROR", message: e instanceof Error ? e.message : String(e) }));
      return c.json(error("upstream_error", "Model backend request failed"), 502);
    }
  });

  return app;
}
