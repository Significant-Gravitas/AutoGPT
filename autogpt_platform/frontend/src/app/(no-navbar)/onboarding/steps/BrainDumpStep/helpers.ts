import type { HTTPValidationError } from "@/app/api/__generated__/models/hTTPValidationError";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import type { OrbScreen } from "./components/MicButton";
import type { ScreenState } from "./useBrainDumpStep";

// Pure helpers for the brain dump. Timings and copy live here so the hook
// stays about behaviour and the component stays about pixels.

// A depth meter, not a limit: the ring fills toward 3:00 and then holds.
// Nothing stops at 3:00 — the longer someone talks, the better the dump.

// Recording keeps going far past the ring: 30 min is the hard stop, and
// even then everything captured is kept.
export const HARD_STOP_SECONDS = 1800;

// The backend refuses longer typed text with a 422 (`MAX_TYPED_CHARS` in
// `onboarding_dump/models.py`); a test pins the two together.
export const MAX_TYPED_CHARS = 20_000;
const TYPED_CHARS_LEFT_FROM = 2_000;
export const TYPED_TOO_LONG_COPY = `That's over the ${formatCount(MAX_TYPED_CHARS)}-character limit; trim it and try again.`;

// Every 3s the recorder hands us a chunk to persist. Small enough that a
// crash costs almost nothing, large enough not to thrash IndexedDB.
export const TIMESLICE_MS = 3000;

// How often the take's metadata row is refreshed with the duration so
// far. A crash never reaches `stop()`, so without this the recovery
// prompt would offer back a take it believes is 0:00 long.
export const META_REFRESH_SECONDS = 5;

export const SILENCE_NUDGE_SECONDS = 5;

// Waveform peak (0-127 either side of the 128 midpoint) that counts as
// somebody talking rather than a quiet room.
export const SPEECH_PEAK_THRESHOLD = 12;
export const SILENCE_NUDGE_COPY =
  "Start anywhere. What did you do yesterday that bored you?";

const ENCOURAGEMENT_COPY = [
  "Keep going, this is gold",
  "The more you share, the sharper Otto gets",
  "You're building Otto's memory right now",
  "You're doing great. Keep going",
  "Every detail makes Otto more useful",
  "Share whatever comes to mind next",
] as const;

const ENCOURAGEMENT_MILESTONES = [
  20, 40, 60, 80, 100, 120, 150, 180, 210, 240, 270, 300, 330, 360,
] as const;

const ENCOURAGEMENTS = ENCOURAGEMENT_MILESTONES.map((atSeconds, index) => ({
  atSeconds,
  text: ENCOURAGEMENT_COPY[index % ENCOURAGEMENT_COPY.length],
}));

const ENCOURAGEMENT_VISIBLE_SECONDS = 6;
const DURATION_GUIDANCE_START_SECONDS = 4;
const DURATION_GUIDANCE_END_SECONDS = 10;
export const DURATION_GUIDANCE_COPY =
  "Even a minute helps. Keep going if you have more.";

export function encouragementAt(elapsedSeconds: number): string | null {
  const active = ENCOURAGEMENTS.find(
    (line) =>
      elapsedSeconds >= line.atSeconds &&
      elapsedSeconds < line.atSeconds + ENCOURAGEMENT_VISIBLE_SECONDS,
  );
  return active?.text ?? null;
}

export function recordingFeedbackAt(elapsedSeconds: number) {
  const encouragement = encouragementAt(elapsedSeconds);
  if (encouragement) return encouragement;
  if (
    elapsedSeconds >= DURATION_GUIDANCE_START_SECONDS &&
    elapsedSeconds < DURATION_GUIDANCE_END_SECONDS
  ) {
    return DURATION_GUIDANCE_COPY;
  }
  return null;
}

export function formatElapsed(totalSeconds: number) {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}

// Chrome/Edge give us webm/opus; Safari only offers mp4. Mirrors the
// existing copilot recorder so both paths hit the same server allowlist.
export function pickMimeType() {
  if (typeof MediaRecorder === "undefined") return "audio/webm";
  return MediaRecorder.isTypeSupported("audio/webm")
    ? "audio/webm"
    : "audio/mp4";
}

export function newRecordingId() {
  return crypto.randomUUID();
}

export const BRAIN_DUMP_HEADLINE = "Talk to me about your work";

// The backend's quality gate rejects dumps with these codes when the
// transcription succeeded but carried nothing to personalize from —
// silence, filler, or an STT hallucination. Distinct from a transcription
// failure: the take went through fine, it just didn't say enough.
// `quality_check_unavailable` is left out on purpose: the judge never ruled,
// so the take gets the failure screen, whose retry can still pass.
const INSUFFICIENT_DUMP_ERROR_CODES = [
  "no_usable_speech",
  "insufficient_content",
] as const;

export function isInsufficientDump(errorCode: unknown) {
  return (
    typeof errorCode === "string" &&
    INSUFFICIENT_DUMP_ERROR_CODES.some((code) => code === errorCode)
  );
}

// Why a take failed: the backend's `error_code`, the HTTP status a request
// was refused with, or "network" when no answer came back.
export type FailureCause = string | number | null;

const FAILURE_REASONS: Record<string, string> = {
  storage_failed: "We couldn't store your recording on our side.",
  no_audio_received: "Your recording didn't reach us.",
  upload_failed: "Part of your recording didn't reach us.",
  transcription_failed: "We couldn't turn your recording into text.",
  transcription_unavailable: "Our speech-to-text service is unavailable.",
  quality_check_unavailable: "We couldn't check what you told us just now.",
  superseded:
    "Something newer you sent, maybe from another tab, replaced this.",
  "429": "You've tried a few times already. Wait a bit first.",
  "503": "Personalization is unavailable for a moment.",
  network: "We couldn't reach the server. Check your connection.",
};

export function failureReason(cause: FailureCause) {
  return cause === null ? null : (FAILURE_REASONS[String(cause)] ?? null);
}

export function failureCauseOf(error: unknown): FailureCause {
  return error instanceof ApiError ? error.status : "network";
}

export function typedCharsLeftHint(length: number) {
  const left = Math.max(0, MAX_TYPED_CHARS - length);
  if (left > TYPED_CHARS_LEFT_FROM) return null;
  return `${formatCount(left)} ${left === 1 ? "character" : "characters"} left`;
}

export function isTypedTextTooLong(error: unknown) {
  if (!(error instanceof ApiError) || error.status !== 422) return false;
  const detail: HTTPValidationError["detail"] = error.response?.detail;
  return (
    Array.isArray(detail) &&
    detail.some(
      (issue) => issue.type === "string_too_long" && issue.loc.includes("text"),
    )
  );
}

function formatCount(count: number) {
  return count.toLocaleString("en-US");
}

export function toOrbScreen(screen: ScreenState): OrbScreen | null {
  // "insufficient" gets no orb: the orb's failed state retries the same
  // take, and re-submitting a rejected take can only be rejected again.
  // Typing keeps the resting avatar and swaps only the control under it.
  if (screen === "recovery" || screen === "insufficient") return null;
  if (screen === "typing") return "rest";
  return screen;
}
