import { z } from "zod";
import type { SystemOneRequest } from "./types.ts";

export const MAX_CHOICE_OPTIONS = 255;
export const MIN_SCORE_LEVELS = 2;
export const MAX_SCORE_LEVELS = 10;

const json: z.ZodType = z.lazy(() =>
  z.union([z.string(), z.number(), z.boolean(), z.null(), z.array(json), z.record(z.string(), json)]),
);

const noul = z.strictObject({
  type: z.literal("noul"),
  instructions: json,
  criteria: z.strictObject({ true: z.string().optional(), false: z.string().optional() }).optional(),
});

const choice = z.strictObject({
  type: z.literal("choice"),
  instructions: json,
  criteria: z
    .record(z.string().min(1), z.string().nullable())
    .refine((c) => Object.keys(c).length >= 2, "choice needs at least 2 options")
    .refine((c) => Object.keys(c).length <= MAX_CHOICE_OPTIONS, `choice supports at most ${MAX_CHOICE_OPTIONS} options`),
});

const score = z.strictObject({
  type: z.literal("score"),
  instructions: json,
  criteria: z.array(z.string()).min(MIN_SCORE_LEVELS).max(MAX_SCORE_LEVELS),
});

const request = z.object({
  state: json.refine((s) => s !== undefined, "state is required"),
  model: z.string().optional(),
  questions: z
    .record(z.string().min(1), z.discriminatedUnion("type", [noul, choice, score]))
    .refine((q) => Object.keys(q).length >= 1, "at least one question is required"),
});

export type ValidationIssue = { path: string; message: string };

export function parseRequest(body: unknown): { ok: true; value: SystemOneRequest } | { ok: false; issues: ValidationIssue[] } {
  const result = request.safeParse(body);
  if (result.success) return { ok: true, value: result.data as SystemOneRequest };
  return {
    ok: false,
    issues: result.error.issues.map((i) => ({ path: i.path.join("."), message: i.message })),
  };
}
