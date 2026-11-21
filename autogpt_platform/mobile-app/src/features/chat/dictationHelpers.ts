export const MAX_DICTATION_MS = 2 * 60 * 1000;

export const MIC_DENIED_MESSAGE = "Allow microphone access in Settings to dictate.";
export const DICTATION_UNAVAILABLE_MESSAGE = "Dictation isn't available right now.";
export const NOT_CAUGHT_MESSAGE = "Couldn't catch that. Try again.";

/** What dictated words follow: the typed text, with a space when it needs one. */
export function dictationLead(current: string) {
  if (!current.trim()) return "";
  return /\s$/.test(current) ? current : `${current} `;
}

/** Dictated text goes after what is already typed. */
export function appendDictation(current: string, dictated: string) {
  const addition = dictated.trim();
  if (!addition) return current;
  return dictationLead(current) + addition;
}

/** `0:04`, `1:30`. */
export function elapsedLabel(ms: number) {
  const seconds = Math.max(0, Math.floor(ms / 1000));
  const rest = seconds % 60;
  return `${Math.floor(seconds / 60)}:${rest < 10 ? "0" : ""}${rest}`;
}

/**
 * The line for a failed transcription, by the route's status (null when the
 * request never got an answer). The route answers 401 both when the server has
 * no transcription key and when the cookie is stale; neither is the user's to
 * fix from here, so neither signs them out.
 */
export function transcribeFailureMessage(status: number | null) {
  if (status === 401 || (status !== null && status >= 500)) {
    return DICTATION_UNAVAILABLE_MESSAGE;
  }
  return NOT_CAUGHT_MESSAGE;
}

const QUIET_DB = -60;

/** The recorder's metering (dBFS, about -160 to 0) as 0 to 1. */
export function meterLevel(db: number | undefined) {
  if (db === undefined || Number.isNaN(db)) return 0;
  return Math.min(1, Math.max(0, (db - QUIET_DB) / -QUIET_DB));
}
