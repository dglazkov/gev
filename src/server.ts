import { serve } from "@hono/node-server";
import { createApp } from "./app.ts";
import { apiKeysFromEnv, backendFromEnv, engineOptionsFromEnv } from "./config.ts";

const backend = backendFromEnv();
const app = createApp({ backend, engine: engineOptionsFromEnv(), apiKeys: apiKeysFromEnv() });
const port = Number(process.env.PORT ?? 8080);

const server = serve({ fetch: app.fetch, port }, () => console.log(`gev listening on :${port} (model: ${backend.model})`));

// Cloud Run sends SIGTERM before stopping an instance; finish in-flight requests first.
process.on("SIGTERM", () => server.close(() => process.exit(0)));
