// Replays recorded jev exchanges (see bench/record.ts) against gev and reports how closely
// gev's decisions track jev's, and how long they take.
//
//   node --env-file=.env bench/compare.ts <suite> [--concurrency N] [-v]
//
// Needs GEV_URL and GEV_API_KEY. Writes gev's answers to bench/results/<suite>.<strategy>.json.

import { mkdir, readFile, readdir, writeFile } from "node:fs/promises";
import type { Answer, SystemOneRequest, SystemOneResponse } from "../src/types.ts";

type Fixture = { request: SystemOneRequest; jev: { ms: number; response: SystemOneResponse } };
type Row = { fixture: string; state: string; ms: number; error?: string; response?: SystemOneResponse };

const args = process.argv.slice(2);
const suite = args.find((a) => !a.startsWith("-") && args[args.indexOf(a) - 1] !== "--concurrency");
const concurrency = Number(args[args.indexOf("--concurrency") + 1]) || 1;
const verbose = args.includes("-v");
const { GEV_URL, GEV_API_KEY } = process.env;
if (!suite || !GEV_URL) throw new Error("usage: GEV_URL=… GEV_API_KEY=… compare.ts <suite> [--concurrency N] [-v]");

const dir = new URL(`./fixtures/${suite}/`, import.meta.url);
const names = (await readdir(dir)).filter((n) => n.endsWith(".json")).sort();
const fixtures: Fixture[] = await Promise.all(names.map(async (n) => JSON.parse(await readFile(new URL(n, dir), "utf8"))));

async function ask(fixture: Fixture, name: string): Promise<Row> {
  const started = performance.now();
  const row = { fixture: name, state: JSON.stringify(fixture.request.state).slice(0, 90) };
  try {
    const response = await fetch(`${GEV_URL}/v1/systemone`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${GEV_API_KEY}` },
      body: JSON.stringify(fixture.request),
    });
    const ms = Math.round(performance.now() - started);
    if (!response.ok) return { ...row, ms, error: `${response.status} ${(await response.text()).slice(0, 200)}` };
    return { ...row, ms, response: (await response.json()) as SystemOneResponse };
  } catch (e) {
    return { ...row, ms: Math.round(performance.now() - started), error: String(e) };
  }
}

const rows: Row[] = [];
for (let i = 0; i < fixtures.length; i += concurrency) {
  rows.push(...(await Promise.all(fixtures.slice(i, i + concurrency).map((f, j) => ask(f, names[i + j]!)))));
  process.stdout.write(`\r${rows.length}/${fixtures.length}`);
}
process.stdout.write("\r");

/** Total variation distance between two distributions over the same keys, 0..1. */
const tv = (a: Record<string, number>, b: Record<string, number>) =>
  Object.keys(a).reduce((sum, k) => sum + Math.abs(a[k]! - (b[k] ?? 0)), 0) / 2;
const mean = (xs: number[]) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : NaN);
const quantile = (xs: number[], q: number) => [...xs].sort((a, b) => a - b)[Math.min(xs.length - 1, Math.floor(q * xs.length))] ?? NaN;

type Tally = { n: number; agree: number; distance: number[]; misses: string[] };
const tally: Record<"choice" | "score" | "noul", Tally> = {
  choice: { n: 0, agree: 0, distance: [], misses: [] },
  score: { n: 0, agree: 0, distance: [], misses: [] },
  noul: { n: 0, agree: 0, distance: [], misses: [] },
};
const perQuestion: Record<string, { n: number; agree: number }> = {};

rows.forEach((row, i) => {
  if (!row.response) return;
  for (const [id, theirs] of Object.entries(fixtures[i]!.jev.response.answers)) {
    const ours = row.response.answers[id] as Answer | undefined;
    if (!ours || ours.type !== theirs.type) continue;
    const t = tally[theirs.type];
    let agree: boolean;
    let shown: string;
    if (theirs.type === "choice" && ours.type === "choice") {
      agree = ours.choice === theirs.choice;
      t.distance.push(tv(theirs.probabilities, ours.probabilities));
      shown = `jev ${theirs.choice}(${theirs.probabilities[theirs.choice]?.toFixed(2)})  gev ${ours.choice}(${ours.probabilities[ours.choice]?.toFixed(2)})`;
    } else if (theirs.type === "score" && ours.type === "score") {
      const top = Object.keys(theirs.legend).length - 1;
      agree = Math.round(ours.score) === Math.round(theirs.score);
      t.distance.push(Math.abs(ours.score - theirs.score) / top);
      shown = `jev ${theirs.score.toFixed(2)}  gev ${ours.score.toFixed(2)}  (0..${top})`;
    } else if (theirs.type === "noul" && ours.type === "noul") {
      agree = ours.noul >= 0.5 === theirs.noul >= 0.5;
      t.distance.push(Math.abs(ours.noul - theirs.noul));
      shown = `jev ${theirs.noul.toFixed(2)}  gev ${ours.noul.toFixed(2)}`;
    } else continue;
    t.n++;
    const q = (perQuestion[id] ??= { n: 0, agree: 0 });
    q.n++;
    if (agree) {
      t.agree++;
      q.agree++;
    } else t.misses.push(`${id.padEnd(18)} ${shown.padEnd(52)} ${row.state}`);
  }
});

const ok = rows.filter((r) => r.response);
console.log(`\n== ${suite}: ${fixtures.length} requests × ${Object.keys(fixtures[0]!.request.questions).length} questions, concurrency ${concurrency} ==`);
console.log(`gev: ${ok[0]?.response?.model ?? "?"}   errors: ${rows.length - ok.length}`);
console.table({
  jev: { "median ms": quantile(fixtures.map((f) => f.jev.ms), 0.5), "p90 ms": quantile(fixtures.map((f) => f.jev.ms), 0.9), "max ms": Math.max(...fixtures.map((f) => f.jev.ms)) },
  gev: { "median ms": quantile(ok.map((r) => r.ms), 0.5), "p90 ms": quantile(ok.map((r) => r.ms), 0.9), "max ms": Math.max(...ok.map((r) => r.ms)) },
});
console.log("Agreement with jev (same top choice / same rounded score / same side of 0.5), and mean distance (TV / normalized |Δscore| / |Δp|):");
console.table(Object.fromEntries(Object.entries(tally).filter(([, t]) => t.n).map(([type, t]) => [type, { agree: `${t.agree}/${t.n}`, "%": Math.round((100 * t.agree) / t.n), "mean distance": Number(mean(t.distance).toFixed(3)) }])));
const worst = Object.entries(perQuestion).filter(([, q]) => q.agree < q.n).sort((a, b) => a[1].agree / a[1].n - b[1].agree / b[1].n);
console.log("Questions with disagreements:", worst.map(([id, q]) => `${id} ${q.agree}/${q.n}`).join(", ") || "none");
rows.filter((r) => r.error).slice(0, 5).forEach((r) => console.log(`ERROR ${r.fixture}: ${r.error}`));
if (verbose) for (const t of Object.values(tally)) t.misses.forEach((m) => console.log(`  ${m}`));

await mkdir(new URL("./results/", import.meta.url), { recursive: true });
const strategy = ok[0]?.response?.gev?.strategy ?? "isolated";
const calls = ok.map((r) => r.response?.gev?.model_calls ?? NaN);
console.log(`strategy: ${strategy}   model calls per request: median ${quantile(calls, 0.5)}   repaired answers: ${ok.reduce((n, r) => n + (r.response?.gev?.repaired ?? 0), 0)}`);
await writeFile(new URL(`./results/${suite}.${strategy}.json`, import.meta.url), JSON.stringify(rows, null, 1));
