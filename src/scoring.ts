// Turns first-token logprobs into probability distributions over answer labels.

export type TokenLogprob = { token: string; logprob: number };

// Tokenizers decorate word-initial tokens ("▁A", "ĠA", " A"); fold those and case
// so every surface form of a label pools into the same bucket.
function normalizeToken(token: string): string {
  return token.replace(/^[\s▁Ġ]+/, "").trim().toLowerCase();
}

/**
 * Collapse the top-k candidates for the first generated token into a normalized
 * distribution over `labels`. Tokens that aren't labels are discarded (that mass
 * is the model going off-script, and renormalizing removes it).
 *
 * A label missing from the top-k still has nonzero probability — we only know it
 * is below the smallest candidate we can see, so it gets half of that as a floor.
 */
export function labelDistribution(top: TokenLogprob[], labels: string[]): number[] {
  const mass = new Map<string, number>(labels.map((l) => [l.toLowerCase(), 0]));
  let smallest = Infinity;
  for (const { token, logprob } of top) {
    const p = Math.exp(logprob);
    if (p < smallest) smallest = p;
    const key = normalizeToken(token);
    if (mass.has(key)) mass.set(key, mass.get(key)! + p);
  }
  const floor = Number.isFinite(smallest) ? smallest / 2 : 1;
  const raw = labels.map((l) => mass.get(l.toLowerCase()) || floor);
  return normalize(raw);
}

export function normalize(values: number[]): number[] {
  const total = values.reduce((a, b) => a + b, 0);
  if (total <= 0) return values.map(() => 1 / values.length);
  return values.map((v) => v / total);
}

/** 1 − normalized entropy: 0 for a uniform distribution, 1 for a one-hot. */
export function confidence(probabilities: number[]): number {
  if (probabilities.length < 2) return 1;
  let entropy = 0;
  for (const p of probabilities) if (p > 0) entropy -= p * Math.log(p);
  return clamp01(1 - entropy / Math.log(probabilities.length));
}

/** Expected zero-indexed level, e.g. [0.05, 0.3, 0.65] → 1.6. */
export function expectedLevel(probabilities: number[]): number {
  return probabilities.reduce((sum, p, i) => sum + p * i, 0);
}

export function argmax(values: number[]): number {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i]! > values[best]!) best = i;
  return best;
}

export function mean(rows: number[][]): number[] {
  const first = rows[0] ?? [];
  return first.map((_, i) => rows.reduce((sum, row) => sum + row[i]!, 0) / rows.length);
}

function clamp01(x: number): number {
  return Math.min(1, Math.max(0, x));
}

export function round(x: number, digits = 4): number {
  const f = 10 ** digits;
  return Math.round(x * f) / f;
}
