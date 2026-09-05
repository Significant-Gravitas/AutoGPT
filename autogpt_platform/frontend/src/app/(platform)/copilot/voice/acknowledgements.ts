/**
 * One canned phrase, spoken the instant the user stops talking.
 *
 * AutoPilot's first token lands a median 13.9 s later, so nothing the model
 * says can fill that gap — and a turn queued behind another returns an empty
 * stream, where the model never speaks at all.
 *
 * Every phrase has to work after a question and after an instruction alike:
 * the transcript that would tell them apart is still a second away when this
 * is chosen.
 */
const PHRASES = [
  "One moment.",
  "Sure thing.",
  "Okay.",
  "Right.",
  "Alright.",
  "Got it.",
  "Mm-hm.",
  "Hang on a sec.",
  "Just a moment.",
  "Give me a second.",
  "Two seconds.",
  "Bear with me.",
  "Let me check.",
  "Let me see.",
  "Let me look that up.",
  "Let me dig into that.",
  "Checking now.",
  "Looking into that.",
  "I'll have a look.",
  "On it.",
  "Working on it.",
  "Sure, one sec.",
];

export function pickAcknowledgement(
  previous: string | null = null,
  random: () => number = Math.random,
): string {
  const candidates = PHRASES.filter((phrase) => phrase !== previous);
  return candidates[Math.floor(random() * candidates.length)];
}
