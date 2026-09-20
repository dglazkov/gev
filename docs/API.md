# Using the gev API

gev answers typed questions about a piece of state, with probabilities. It generates no text. You send
**state** (any JSON) and a set of **questions**; you get back one typed answer per question, which your
code can threshold, rank, or use to defer to a human. The request and response shapes are
[jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)'s `POST /v1/systemone`.

- **Base URL:** `https://gev-huio5ftumq-uc.a.run.app`
- **Auth:** `Authorization: Bearer <your key>` (ask Dimitri for one; it looks like `sk-gev-<name>-…`)
- **Try it in a browser:** open the base URL, paste the key, edit the example, press Decide.

Experimental, one GPU shared by everyone, no uptime promise. Keep the key out of public repos and
client-side code that ships to strangers.

## A request

```bash
export GEV_URL=https://gev-huio5ftumq-uc.a.run.app
export GEV_API_KEY=sk-gev-…

curl -s "$GEV_URL/v1/systemone" -H "authorization: Bearer $GEV_API_KEY" -H 'content-type: application/json' -d '{
  "state": "I was charged twice and my demo is in two hours. Fix this NOW.",
  "questions": {
    "team":        { "type": "choice", "instructions": "Which team should handle this?",
                     "criteria": { "billing": "Payment issues", "technical": "Bugs", "sales": null } },
    "urgent":      { "type": "noul",   "instructions": "The message conveys urgency" },
    "frustration": { "type": "score",  "instructions": "How frustrated is the customer?",
                     "criteria": ["Calm", "Frustrated but civil", "Very angry"] }
  }
}'
```

```json
{
  "model": "RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic",
  "answers": {
    "team":        { "type": "choice", "choice": "billing", "probabilities": { "billing": 0.97, "technical": 0.02, "sales": 0.01 }, "confidence": 0.86 },
    "urgent":      { "type": "noul", "noul": 0.99 },
    "frustration": { "type": "score", "score": 1.8, "legend": { "0": "Calm", "1": "Frustrated but civil", "2": "Very angry" },
                     "probabilities": { "0": 0.01, "1": 0.18, "2": 0.81 }, "confidence": 0.52 }
  },
  "usage": { "input_tokens": 612, "output_tokens": 3 },
  "gev": { "strategy": "scored", "model_calls": 1, "repaired": 0, "ms": 95, "model_ms": 88 }
}
```

(Illustrative numbers.)

## The request

| Field | | |
| --- | --- | --- |
| `state` | required | Any JSON: a string, an object, a conversation. What the questions are about. |
| `questions` | required | An object of one or more questions. The keys are yours and come back as the keys of `answers`. |
| `model` | optional | Accepted for jev compatibility and ignored. |

Every question has a `type`, `instructions` (a string, or any JSON), and `criteria`:

| `type` | `criteria` | Answer |
| --- | --- | --- |
| `noul` | optional `{ "true": "…", "false": "…" }` describing what yes and no mean | `noul`: the probability, 0 to 1, that the statement in `instructions` is true |
| `choice` | required: an object of 2 to 255 options, `"name": "description"` or `"name": null` | `choice`: the most likely option's name. `probabilities`: one per option, summing to 1 |
| `score` | required: an array of 2 to 10 level descriptions, lowest first | `score`: the expected level, zero-indexed and fractional (1.8 of 0–2). `legend`: index → description. `probabilities`: one per level |

`confidence` (choice and score) is 1 minus the normalized entropy of the distribution: 0 means uniform,
1 means certain. Unknown fields inside a question are rejected.

`gev` in the response is not part of jev's API: `ms` is the time spent inside the server and `model_ms`
the part of it spent in the model, which helps tell server time from network time.

## Getting the most out of it

- **Ask everything in one request.** All questions of a request are answered in a single model pass, so
  20 questions cost about the same time as one: roughly 100 ms server-side for 18–28 questions. Twenty
  separate requests cost twenty round trips.
- **Questions are answered independently.** One answer never sees another, so a question can't refer to
  "the team chosen above".
- **Keep question text stable between requests.** The instructions and criteria are cached on the
  server; only the state is read fresh. Rewording questions on every call is slower.
- **Write `noul` instructions as statements**, not questions: "The message conveys urgency".
- **Treat probabilities as rankings, not calibrated odds.** They are a language model's token
  probabilities, softened. They are good for ordering and thresholding; measure on your own data before
  trusting 0.9 to mean 90%.
- The first request after a quiet spell takes about two seconds while the API wakes up.

## With the TypeSafe SDK

gev speaks jev's wire format, so a jev client works unmodified: `@typesafe-ai/sdk` reads its base URL
from `TYPESAFE_BASE_URL`. Set that to the base URL above and give the SDK the gev key in place of a jev
key.

## Errors

Errors look like `{ "error": { "type": "…", "message": "…" } }`.

| Status | `type` | |
| --- | --- | --- |
| `401` | `unauthorized` | Missing or wrong key |
| `422` | `invalid_request` | Body isn't JSON, or failed validation; `error.issues` lists `{ path, message }` for each problem |
| `429` | `rate_limited` | The model server is saturated. Retry with exponential backoff |
| `502` | `upstream_error` | The model server failed. Retry with exponential backoff |

CORS is open on `/v1/*`, so a browser app can call the API directly, which also exposes the key to
whoever opens that app.

How it works inside, and how to run your own: [README](../README.md).
