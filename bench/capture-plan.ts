// Sends jev2ui's real "plan the screen" request (its 28-question tree) for a set of prompts,
// so the recording proxy captures exactly what the app asks. Run from the jev2ui checkout,
// which supplies the question builders and their dependencies:
//
//   cd ../jev2ui && TYPESAFE_BASE_URL=http://localhost:8790 npx tsx ../gev/bench/capture-plan.ts

const jev2ui = process.cwd();
const { planQuestions } = await import(`${jev2ui}/src/server/mock/plan.ts`);
const { askJev } = await import(`${jev2ui}/src/server/models.ts`);

// jev2ui's own eval prompts (src/eval.ts).
const PROMPTS = [
  "Delete my account and all of its data",
  "What happens to my data if I delete my account?",
  "Am I going to be surprised by my electricity bill this month?",
  "Send $50 to Alex",
  "My car won't start, what do I do?",
  "Sign-up form for a weekend pottery workshop",
  "Find me three Italian restaurants near downtown Seattle for tonight",
  "How do I make sourdough starter from scratch?",
  "Tell me about the Golden Gate Bridge",
  "Settings for notification preferences in a chat app",
  "Pick a movie for family night",
  "Checkout screen for a sneaker store with order summary",
  "Home energy dashboard showing today's usage",
];

const questions = planQuestions();
console.log(`${PROMPTS.length} prompts × ${Object.keys(questions).length} questions`);
for (const screen of PROMPTS) {
  const { ms, answers } = await askJev({ screen }, questions);
  console.log(`${Math.round(ms)} ms  ${answers.archetype?.choice ?? "?"}  ${screen}`);
}
