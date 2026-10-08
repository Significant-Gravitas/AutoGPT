/** Where the user is in an onboarding card, kept outside React state.
 *
 * The card's row is re-keyed when the kickoff turn settles (streamed AI SDK
 * ids give way to the saved ``<sessionId>-seq-N`` ids), which remounts the
 * card and would otherwise reset it to question 1 with empty answers. The
 * tool call id is the same on both copies, so progress keyed by it survives
 * the swap — and, since it lives in sessionStorage, a reload mid-card too. */

export interface OnboardingProgress {
  step: number;
  answers: Record<string, string>;
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

function toProgress(value: unknown): OnboardingProgress | null {
  if (!value || typeof value !== "object" || Array.isArray(value)) return null;
  const record = value as Record<string, unknown>;
  const step =
    typeof record.step === "number" && Number.isInteger(record.step)
      ? Math.max(record.step, 0)
      : 0;
  const answers: Record<string, string> = {};
  if (
    record.answers &&
    typeof record.answers === "object" &&
    !Array.isArray(record.answers)
  ) {
    for (const [keyword, answer] of Object.entries(record.answers)) {
      if (typeof answer === "string") answers[keyword] = answer;
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
