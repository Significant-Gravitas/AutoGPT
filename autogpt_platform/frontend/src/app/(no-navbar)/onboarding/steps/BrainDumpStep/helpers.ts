// Pure helpers for the brain dump. Timings and copy live here so the hook
// stays about behaviour and the component stays about pixels.

// A depth meter, not a limit: the ring fills toward 3:00 and then holds.
// Nothing stops at 3:00 — the longer someone talks, the better the dump.
export const RING_TARGET_SECONDS = 180;

// Recording keeps going far past the ring: 30 min is the hard stop, and
// even then everything captured is kept.
export const HARD_STOP_SECONDS = 1800;

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
  "The more you share, the sharper AutoPilot gets",
  "You're building AutoPilot's memory right now",
  "You're doing great — keep going",
  "Every detail makes AutoPilot more useful",
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
export const DURATION_GUIDANCE_COPY = "Most people talk for 2 to 3 minutes.";

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

export function ringProgress(elapsedSeconds: number) {
  return Math.min(1, elapsedSeconds / RING_TARGET_SECONDS);
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

export function isPermissionDenied(error: unknown) {
  return error instanceof DOMException && error.name === "NotAllowedError";
}

export function headline(name: string) {
  const trimmed = name.trim();
  return trimmed
    ? `What keeps stealing your week, ${trimmed}?`
    : "What keeps stealing your week?";
}

// The backend's quality gate rejects dumps with these codes when the
// transcription succeeded but carried nothing to personalize from —
// silence, filler, or an STT hallucination. Distinct from a transcription
// failure: the take went through fine, it just didn't say enough.
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
