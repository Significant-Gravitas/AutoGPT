/** Where the user is in an onboarding card, kept outside React state.
 *
 * The card's row is re-keyed when the kickoff turn settles (streamed AI SDK
 * ids give way to the saved ``<sessionId>-seq-N`` ids), which remounts the
 * card and would otherwise reset it to question 1 with empty answers. The
 * tool call id is the same on both copies, so progress keyed by it survives
 * the swap — and, since it lives in sessionStorage, a reload mid-card too. */

import type {
  MultiAnswer,
  QuestionAnswer,
} from "../../tools/clarifying-questions";

export interface OnboardingProgress {
  step: number;
  answers: Record<string, QuestionAnswer>;
}

const KEY_PREFIX = "copilot:expert-onboarding-progress:";

function storageKey(callId: string): string {
  return `${KEY_PREFIX}${callId}`;
}

function getStorage(): Storage | null {
  try {
    return typeof window !== "undefined" ? window.sessionStorage : null;
  } catch {
    return null;
  }
}

function isStringList(value: unknown): value is string[] {
  return (
    Array.isArray(value) && value.every((item) => typeof item === "string")
  );
}

// Progress saved before multi-select questions holds only strings, which
// still read as single answers.
function toAnswer(value: unknown): QuestionAnswer | null {
  if (typeof value === "string") return value;
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Partial<Record<keyof MultiAnswer, unknown>>;
  if (!isStringList(record.selected) || typeof record.custom !== "string") {
    return null;
  }
  return { selected: record.selected, custom: record.custom };
}

function toProgress(value: unknown): OnboardingProgress | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const step =
    typeof record.step === "number" && Number.isInteger(record.step)
      ? Math.max(record.step, 0)
      : 0;
  const answers: Record<string, QuestionAnswer> = {};
  if (
    record.answers &&
    typeof record.answers === "object" &&
    !Array.isArray(record.answers)
  ) {
    for (const [keyword, raw] of Object.entries(record.answers)) {
      const answer = toAnswer(raw);
      if (answer !== null) answers[keyword] = answer;
    }
  }
  return { step, answers };
}

export function readOnboardingProgress(
  callId: string,
): OnboardingProgress | null {
  const storage = getStorage();
  if (!storage) return null;
  try {
    const raw = storage.getItem(storageKey(callId));
    return raw ? toProgress(JSON.parse(raw) as unknown) : null;
  } catch {
    return null;
  }
}

export function writeOnboardingProgress(
  callId: string,
  progress: OnboardingProgress,
): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.setItem(storageKey(callId), JSON.stringify(progress));
  } catch {
    // Quota or privacy-mode failures only cost the restore, not the card.
  }
}

export function clearOnboardingProgress(callId: string): void {
  const storage = getStorage();
  if (!storage) return;
  try {
    storage.removeItem(storageKey(callId));
  } catch {
    // Nothing to recover from: a key that cannot be removed cannot be read
    // either, and the card still renders as history.
  }
}
