// A recording proxy: point a jev client at it, and every /v1/systemone exchange with the
// real jev is saved as a fixture that bench/compare.ts can replay against gev.
//
//   node --env-file=.env bench/record.ts <suite>        # listens on :8790, writes bench/fixtures/<suite>/
//   TYPESAFE_BASE_URL=http://localhost:8790 npm run probe:jtbd    # in the client's repo

import { mkdir, writeFile } from "node:fs/promises";
import { createServer } from "node:http";

const suite = process.argv[2];
if (!suite) throw new Error("usage: record.ts <suite>");
const upstream = process.env.JEV_BASE_URL ?? "https://api.typesafe.ai";
const port = Number(process.env.RECORD_PORT ?? 8790);
const dir = new URL(`./fixtures/${suite}/`, import.meta.url);
await mkdir(dir, { recursive: true });

let count = 0;

createServer(async (req, res) => {
  const chunks: Buffer[] = [];
  for await (const chunk of req) chunks.push(chunk as Buffer);
  const body = Buffer.concat(chunks).toString("utf8");

  const started = performance.now();
  const response = await fetch(upstream + req.url, {
    method: req.method,
    headers: { "content-type": "application/json", authorization: req.headers.authorization ?? "" },
    body: req.method === "POST" ? body : undefined,
  });
  const text = await response.text();
  const ms = Math.round(performance.now() - started);

  if (req.url === "/v1/systemone" && response.ok) {
    const id = String(++count).padStart(3, "0");
    await writeFile(new URL(`${id}.json`, dir), JSON.stringify({ request: JSON.parse(body), jev: { ms, response: JSON.parse(text) } }, null, 1));
    console.log(`${id}  ${ms} ms  ${Object.keys(JSON.parse(body).questions).length} questions`);
  } else {
    console.log(`${req.method} ${req.url} -> ${response.status} (not recorded)`);
  }
  res.writeHead(response.status, { "content-type": "application/json" }).end(text);
}).listen(port, () => console.log(`recording ${suite}: :${port} -> ${upstream}`));
