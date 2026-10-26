import { Key, storage } from "@/services/storage/local-storage";

import type { TurnPause } from "../../turnPause";

const MAX_STOPPED = 50;

/** "Stop here" is the user's call alone, so it lives on this device only. */
export function isPauseStopped(key: string): boolean {
  return readStopped().includes(key);
}

export function rememberStoppedPause(key: string) {
  const kept = readStopped()
    .filter((stopped) => stopped !== key)
    .slice(-(MAX_STOPPED - 1));
  storage.set(Key.COPILOT_STOPPED_PAUSES, JSON.stringify([...kept, key]));
}

function readStopped(): string[] {
  try {
    const value: unknown = JSON.parse(
      storage.get(Key.COPILOT_STOPPED_PAUSES) ?? "[]",
    );
    return Array.isArray(value)
      ? value.filter((key): key is string => typeof key === "string")
      : [];
  } catch {
    return [];
  }
}

// A stop made on a pause still streaming in, which has no stored row yet to
// keep it under: held per session until that row arrives.
const pendingStops = new Map<string, string>();

export function holdPendingStop(sessionId: string, pause: TurnPause) {
  pendingStops.set(sessionId, pauseFingerprint(pause));
}

export function isPendingStop(sessionId: string, pause: TurnPause): boolean {
  return pendingStops.get(sessionId) === pauseFingerprint(pause);
}

/** The pause's stored row arrived: it keeps the stop from now on. */
export function settlePendingStop(sessionId: string, storageKey: string) {
  rememberStoppedPause(storageKey);
  pendingStops.delete(sessionId);
}

export function dropPendingStop(sessionId: string) {
  pendingStops.delete(sessionId);
}

// The same pause live and stored. The API client hands a stored reset over
// as a Date, whose ISO form differs from the wire's, so it compares as an
// instant.
function pauseFingerprint(pause: TurnPause): string {
  return JSON.stringify([
    pause.reason,
    pause.resets_at === null ? null : Date.parse(pause.resets_at),
    pause.steps,
    pause.checkpoint_saved,
    pause.started,
  ]);
}
