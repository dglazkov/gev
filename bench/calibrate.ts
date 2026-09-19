// Finds the temperature (GEV_TEMPERATURE) per question type that brings gev's probabilities closest
// to jev's, which is what jev2ui's thresholds were tuned against. Temperature doesn't change any
// decision, so each type's best value can be read off one sweep. Fitted on the even-numbered
// fixtures of every suite and checked on the odd ones.
//
//   MODEL_URL=https://gev-ar-….run.app MODEL=… TOKEN=$(gcloud auth print-identity-token) \
//     node bench/calibrate.ts [suite …=jtbd plan] [--temperatures 1,1.5,2,3,4,6]
//
// Distance alone is a trap: where gev and jev disagree, hedging always looks closer, so for nouls
// it keeps falling as T grows and everything collapses towards 0.5. So the table also shows
// sharpness (how far from undecided an answer is, 0..1: `confidence` for choices and scores,
// |2p − 1| for nouls) next to jev's, and how often a noul lands on jev's side of jev2ui's
// thresholds (0.5 and 0.6). The honest target is jev's sharpness without losing agreement.
//
// Distance is the one bench/compare.ts reports: total variation for a choice, |Δscore| over the
// scale for a score, |Δp| for a noul. Talks to the model server directly, like model-probe.ts.

import { readFile, readdir } from "node:fs/promises";
import { VllmBackend } from "../src/backends/vllm.ts";
import { DEFAULT_ENGINE_OPTIONS, systemOne } from "../src/engine.ts";
import type { Answer, SystemOneRequest, SystemOneResponse } from "../src/types.ts";

type Fixture = { request: SystemOneRequest; jev: { ms: number; response: SystemOneResponse } };
type Type = Answer["type"];
const TYPES: Type[] = ["choice", "score", "noul"];

const args = process.argv.slice(2);
const at = args.indexOf("--temperatures");
const temperatures = (at >= 0 ? args[at + 1]! : "1,1.5,2,3,4,6").split(",").map(Number);
const suites = args.filter((a, i) => !a.startsWith("-") && (at < 0 || i !== at + 1));
if (suites.length === 0) suites.push("jtbd", "plan");
const { MODEL_URL, TOKEN, MODEL = "google/gemma-4-26B-A4B-it" } = process.env;
if (!MODEL_URL || !TOKEN) throw new Error("MODEL_URL and TOKEN are required");

const fixtures: Fixture[] = [];
for (const suite of suites) {
  const dir = new URL(`./fixtures/${suite}/`, import.meta.url);
  for (const name of (await readdir(dir)).filter((n) => n.endsWith(".json")).sort()) fixtures.push(JSON.parse(await readFile(new URL(name, dir), "utf8")));
}

function distance(theirs: Answer, ours: Answer): number {
  if (theirs.type === "choice" && ours.type === "choice") return Object.keys(theirs.probabilities).reduce((sum, k) => sum + Math.abs(theirs.probabilities[k]! - (ours.probabilities[k] ?? 0)), 0) / 2;
  if (theirs.type === "score" && ours.type === "score") return Math.abs(ours.score - theirs.score) / (Object.keys(theirs.legend).length - 1);
  if (theirs.type === "noul" && ours.type === "noul") return Math.abs(ours.noul - theirs.noul);
  return NaN;
}

const sharpness = (a: Answer) => (a.type === "noul" ? Math.abs(2 * a.noul - 1) : a.confidence);

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length);
const backend = new VllmBackend({ baseUrl: `${MODEL_URL}/v1`, model: MODEL, apiKey: TOKEN });
const table: Record<string, Record<string, string>> = {};
const fit: Record<Type, { temperature: number; distance: number }> = { choice: { temperature: 1, distance: Infinity }, score: { temperature: 1, distance: Infinity }, noul: { temperature: 1, distance: Infinity } };

for (const temperature of temperatures) {
  const options = { ...DEFAULT_ENGINE_OPTIONS, strategy: "scored" as const, order: "state-last" as const, wideChoice: true, temperature: { choice: temperature, score: temperature, noul: temperature } };
  const halves: Record<Type, [number[], number[]]> = { choice: [[], []], score: [[], []], noul: [[], []] };
  const sharp: Record<Type, number[]> = { choice: [], score: [], noul: [] };
  const sameSide = { "0.5": [] as number[], "0.6": [] as number[] };
  for (const [i, fixture] of fixtures.entries()) {
    const { answers } = await systemOne(backend, fixture.request, options);
    for (const [id, theirs] of Object.entries(fixture.jev.response.answers)) {
      const d = answers[id] ? distance(theirs, answers[id]) : NaN;
      if (Number.isNaN(d)) continue;
      const ours = answers[id]!;
      halves[theirs.type][i % 2]!.push(d);
      sharp[theirs.type].push(sharpness(ours));
      if (theirs.type === "noul" && ours.type === "noul") for (const t of [0.5, 0.6]) sameSide[String(t) as "0.5" | "0.6"].push(Number(ours.noul >= t === theirs.noul >= t));
    }
  }
  table[`T=${temperature}`] = {
    ...Object.fromEntries(TYPES.flatMap((type) => [[`${type} fit`, mean(halves[type][0]).toFixed(3)], [`${type} test`, mean(halves[type][1]).toFixed(3)], [`${type} sharp`, mean(sharp[type]).toFixed(2)]])),
    "noul ≥0.5 as jev": `${Math.round(100 * mean(sameSide["0.5"]))}%`,
    "noul ≥0.6 as jev": `${Math.round(100 * mean(sameSide["0.6"]))}%`,
  };
  for (const type of TYPES) if (halves[type][0].length && mean(halves[type][0]) < fit[type].distance) fit[type] = { temperature, distance: mean(halves[type][0]) };
}

const jevSharp = Object.fromEntries(TYPES.map((type) => [type, mean(fixtures.flatMap((f) => Object.values(f.jev.response.answers).filter((a) => a.type === type).map(sharpness))).toFixed(2)]));
console.log(`Mean distance to jev's probabilities, ${fixtures.length} requests (${suites.join(", ")}), ${MODEL}`);
console.log(`jev's own sharpness: ${TYPES.map((type) => `${type} ${jevSharp[type]}`).join(", ")}`);
console.table(table);
console.log(`Closest to jev on the fit half (see the caveat at the top of this file): GEV_TEMPERATURE=${TYPES.map((type) => `${type}=${fit[type].temperature}`).join(",")}`);
