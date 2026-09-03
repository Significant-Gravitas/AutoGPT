/**
 * Canned phrases spoken the instant the user stops talking.
 *
 * AutoPilot's first token lands a median 13.9 s later, so nothing the model
 * says can fill that gap — and a turn queued behind another returns an empty
 * stream, where the model never speaks at all.
 */

export type UtteranceKind = "question" | "request";

/** Fits either register. */
const NEUTRAL = [
  "One moment.",
  "Sure thing.",
  "Okay.",
  "Right.",
  "Got it.",
  "Hang on a sec.",
  "Just a moment.",
  "Bear with me.",
  "Mm-hm.",
];

const FOR_QUESTION = [
  "Let me check.",
  "Let me look that up.",
  "Good question — one sec.",
  "Checking now.",
  "Looking into that.",
  "Let me find out.",
  "I'll have a look.",
  "Let me see.",
];

const FOR_REQUEST = [
  "On it.",
  "I'll take care of that.",
  "Working on it.",
  "Starting on that now.",
  "Sure, doing that now.",
  "Let me get that going.",
  "I'll get that started.",
];

/**
 * @param kind - `null` before the transcript exists, which is the case for
 *   the phrase spoken the moment speech ends.
 */
export function pickAcknowledgement(
  kind: UtteranceKind | null,
  previous: string | null = null,
  random: () => number = Math.random,
): string {
  const pool = kind === null ? NEUTRAL : REGISTERS[kind];
  const candidates = pool.filter((phrase) => phrase !== previous);
  return candidates[Math.floor(random() * candidates.length)];
}

const REGISTERS: Record<UtteranceKind, string[]> = {
  question: FOR_QUESTION,
  request: FOR_REQUEST,
};

export function classifyUtterance(transcript: string): UtteranceKind {
  const text = transcript.trim().toLowerCase();
  if (text.endsWith("?")) return "question";
  const opener = text.split(/[\s,]+/)[0]?.replace(/[^a-z']/g, "") ?? "";
  return QUESTION_OPENERS.has(opener) ? "question" : "request";
}

const QUESTION_OPENERS = new Set([
  "what",
  "why",
  "how",
  "when",
  "where",
  "who",
  "whose",
  "which",
  "is",
  "are",
  "was",
  "were",
  "am",
  "do",
  "does",
  "did",
  "can",
  "could",
  "should",
  "would",
  "will",
  "have",
  "has",
  "may",
  "any",
]);
