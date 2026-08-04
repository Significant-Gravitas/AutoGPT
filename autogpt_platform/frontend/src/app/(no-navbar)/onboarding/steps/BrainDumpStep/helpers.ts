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

// Never any copy that caps effort — "30 seconds is plenty" is the exact
// message this screen exists to avoid sending.
const ENCOURAGEMENTS = [
  { atSeconds: 10, text: "Keep going, this is gold" },
  { atSeconds: 25, text: "The more you share, the sharper AutoPilot gets" },
  { atSeconds: 45, text: "You're building AutoPilot's memory right now" },
] as const;

// After the last line the screen goes quiet — a nag every 20s would turn
// encouragement into pressure.
const ENCOURAGEMENT_VISIBLE_SECONDS = 6;

export function encouragementAt(elapsedSeconds: number): string | null {
  const active = ENCOURAGEMENTS.find(
    (line) =>
      elapsedSeconds >= line.atSeconds &&
      elapsedSeconds < line.atSeconds + ENCOURAGEMENT_VISIBLE_SECONDS,
  );
  return active?.text ?? null;
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
