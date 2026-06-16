import { Key, storage } from "@/services/storage/local-storage";

/**
 * The cloud-fallback consent (WORKFLOW_RECORDING.md §9.1) is remembered
 * per *kind* of recording — not global, not per-session. A "kind" is a
 * stable descriptor of the recording (in v1: the interpretation route plus
 * the dominant app), so a remembered "yes" for "fill this CRM form" doesn't
 * silently extend to an unrelated recording, and we don't re-interrupt for
 * the same kind. Clearing the stored set (a settings action) revokes all
 * remembered choices.
 */

function readKinds(): string[] {
  const raw = storage.get(Key.COPILOT_RECORDING_CLOUD_CONSENT_KINDS);
  if (!raw) return [];
  try {
    const parsed = JSON.parse(raw);
    if (Array.isArray(parsed)) {
      return parsed.filter((x): x is string => typeof x === "string");
    }
  } catch {
    /* corrupted — treat as empty */
  }
  return [];
}

export function hasRememberedRecordingConsent(kind: string): boolean {
  return readKinds().includes(kind);
}

export function rememberRecordingConsent(kind: string): void {
  const kinds = new Set(readKinds());
  kinds.add(kind);
  storage.set(
    Key.COPILOT_RECORDING_CLOUD_CONSENT_KINDS,
    JSON.stringify(Array.from(kinds)),
  );
}

/** Revoke all remembered cloud-fallback consents (settings action). */
export function revokeAllRecordingConsents(): void {
  storage.clean(Key.COPILOT_RECORDING_CLOUD_CONSENT_KINDS);
}

/**
 * Derive a stable per-kind key for a recording. The cloud-fallback dialog
 * only ever appears for the `screenshots_to_cloud` route (§3.1/§9.1), so
 * the route is always part of the key; the app scopes it so different apps
 * don't share a remembered yes.
 */
export function recordingKind(args: {
  interpretationRoute: string;
  app?: string | null;
}): string {
  const app = (args.app ?? "any").toLowerCase().replace(/\s+/g, "_");
  return `${args.interpretationRoute}:${app}`;
}
