// Wire types for the jev-compatible API (POST /v1/systemone).

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

export type NoulQuestion = {
  type: "noul";
  instructions: Json;
  criteria?: { true?: string; false?: string };
};

export type ChoiceQuestion = {
  type: "choice";
  instructions: Json;
  criteria: Record<string, string | null>;
};

export type ScoreQuestion = {
  type: "score";
  instructions: Json;
  criteria: string[];
};

export type Question = NoulQuestion | ChoiceQuestion | ScoreQuestion;

export type SystemOneRequest = {
  state: Json;
  model?: string;
  questions: Record<string, Question>;
};

export type NoulAnswer = { type: "noul"; noul: number };

export type ChoiceAnswer = {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
};

export type ScoreAnswer = {
  type: "score";
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
};

export type Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer;

export type Usage = { input_tokens: number; output_tokens: number };

export type SystemOneResponse = {
  model: string;
  answers: Record<string, Answer>;
  usage: Usage;
  /** Not part of jev's API: how gev produced these answers. */
  gev?: {
    strategy: "isolated" | "packed" | "scored";
    model_calls: number;
    repaired: number;
    /** Wall time inside gev, and the part of it spent waiting on the model server. */
    ms: number;
    model_ms: number;
  };
};
