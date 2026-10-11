"use client";

import { useSyncExternalStore } from "react";

// Playwright builds only (``NEXT_PUBLIC_PW_TEST=true``): a journey writes a
// ``{ "<flag-key>": value }`` map under this key before the app loads, so one
// bundle can serve both the self-hosted defaults and the hosted flag values
// without a rebuild. Any other build never reads it.
export const PW_FLAG_OVERRIDES_STORAGE_KEY = "autogpt-pw-flag-overrides";

type PwFlagOverrides = Record<string, unknown>;

const NO_OVERRIDES: PwFlagOverrides = {};
const parsedOverrides = new Map<string, PwFlagOverrides>();

function isPwBuild() {
  return process.env.NEXT_PUBLIC_PW_TEST === "true";
}

function subscribe() {
  return () => {};
}

function readStoredOverrides() {
  if (!isPwBuild()) return "";
  try {
    return window.localStorage.getItem(PW_FLAG_OVERRIDES_STORAGE_KEY) ?? "";
  } catch {
    return "";
  }
}

// Outside a Playwright build both snapshots agree, so hydration doesn't
// re-render every flag consumer for nothing.
function readServerSnapshot() {
  return isPwBuild() ? null : "";
}

/**
 * The Playwright flag overrides, or ``null`` while the page is hydrating and
 * the browser's storage can't be read yet.
 */
export function usePwFlagOverrides(): PwFlagOverrides | null {
  const raw = useSyncExternalStore(
    subscribe,
    readStoredOverrides,
    readServerSnapshot,
  );
  if (raw === null) return null;
  return parseOverrides(raw);
}

function parseOverrides(raw: string): PwFlagOverrides {
  if (!raw) return NO_OVERRIDES;
  const cached = parsedOverrides.get(raw);
  if (cached) return cached;
  let parsed: PwFlagOverrides = NO_OVERRIDES;
  try {
    const value: unknown = JSON.parse(raw);
    if (value && typeof value === "object" && !Array.isArray(value)) {
      parsed = value as PwFlagOverrides;
    }
  } catch {
    parsed = NO_OVERRIDES;
  }
  parsedOverrides.set(raw, parsed);
  return parsed;
}
