// Checks a deployed gev against issue #1: a choice with more than 16 names that share a first token.
// Before the fix this request never returned and the API process died of heap exhaustion, taking every
// caller on that instance with it. Afterwards, one ordinary request shows the API is still answering.
//
//   node --env-file=.env bench/wide-choice-probe.ts
//
// Needs GEV_URL and GEV_API_KEY. Point it at the live gev only when that is what you mean to test:
// on a gev without the fix, this crashes it.

const { GEV_URL, GEV_API_KEY } = process.env;
if (!GEV_URL || !GEV_API_KEY) throw new Error("usage: GEV_URL=… GEV_API_KEY=… wide-choice-probe.ts");

async function ask(label: string, body: unknown): Promise<boolean> {
  const started = performance.now();
  try {
    const response = await fetch(`${GEV_URL}/v1/systemone`, {
      method: "POST",
      headers: { "content-type": "application/json", authorization: `Bearer ${GEV_API_KEY}` },
      body: JSON.stringify(body),
      // The crash took ~90 s to kill the process; a healthy answer takes well under a second.
      signal: AbortSignal.timeout(30_000),
    });
    const ms = Math.round(performance.now() - started);
    const json = (await response.json()) as { answers?: Record<string, { choice?: string; noul?: number }> };
    console.log(`${label}: ${response.status} in ${ms} ms, answer ${JSON.stringify(json.answers?.q)}`);
    return response.ok;
  } catch (e) {
    console.log(`${label}: failed after ${Math.round(performance.now() - started)} ms: ${e}`);
    return false;
  }
}

// "item1a"…"item1t": twenty names with the same first token and no separator to rotate on.
const criteria = Object.fromEntries(Array.from({ length: 20 }, (_, i) => [`item1${String.fromCharCode(97 + i)}`, null]));
const wide = await ask("20 names, one first token", {
  state: "The user asked for item1r.",
  questions: { q: { type: "choice", instructions: "Which item did the user ask for?", criteria } },
});
const after = await ask("ordinary request after it", {
  state: "It is raining.",
  questions: { q: { type: "noul", instructions: "It is raining." } },
});
console.log(wide && after ? "OK: answered, and the API is still up" : "FAILED");
process.exitCode = wide && after ? 0 : 1;
