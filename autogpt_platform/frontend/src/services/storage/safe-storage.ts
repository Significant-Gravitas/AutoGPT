import * as Sentry from "@sentry/nextjs";
import { environment } from "../environment";

type Area = "local" | "session";

/**
 * Stand-in used in the browser when the real Web Storage cannot be reached:
 * private mode, third-party storage blocked, or an embedded WebView. Values
 * survive for the life of the tab, which is enough for the preferences we
 * keep here, and nothing throws.
 *
 * Deliberately never used server-side. One node process serves every visitor,
 * so a module-scoped map there would hand one visitor's value to the next one
 * in their HTML.
 */
const memory: Record<Area, Map<string, string>> = {
  local: new Map(),
  session: new Map(),
};

function memoryFor(area: Area): Map<string, string> | null {
  return environment.isServerSide() ? null : memory[area];
}

/**
 * The browser's storage object, or null when there isn't one to talk to.
 * Reading the property is itself enough to throw on some WebViews, and some
 * browsers hand back null rather than throwing.
 */
function nativeStorage(area: Area): Storage | null {
  if (environment.isServerSide()) return null;
  try {
    const store =
      area === "local" ? window.localStorage : window.sessionStorage;
    return store ?? null;
  } catch {
    return null;
  }
}

/**
 * Web Storage that degrades instead of throwing, for a given storage area.
 *
 * Callers get a plain `string | null` and never need to guard for the server
 * or for a browser that refuses storage.
 */
export function createSafeStorage<K extends string>(area: Area) {
  function get(key: K): string | null {
    const store = nativeStorage(area);
    if (!store) return memoryFor(area)?.get(key) ?? null;
    try {
      // Memory is an overlay, not a cache: it only holds keys whose write to
      // the real storage failed, and a successful write clears its entry. So
      // an overlay hit is always newer than what the store has, and it wins —
      // which is what keeps "a set reads back within the tab" true even when
      // the key already had a value the failed write was meant to replace.
      return memoryFor(area)?.get(key) ?? store.getItem(key) ?? null;
    } catch {
      return memoryFor(area)?.get(key) ?? null;
    }
  }

  function set(key: K, value: string): void {
    const store = nativeStorage(area);
    if (!store) {
      memoryFor(area)?.set(key, value);
      return;
    }
    try {
      store.setItem(key, value);
      // The store is now authoritative for this key again; drop any overlay
      // entry left by an earlier failed write so it cannot mask this value.
      memoryFor(area)?.delete(key);
    } catch (error) {
      // The storage object is live, so this is something unexpected
      // (QuotaExceededError and friends) rather than a browser that simply
      // refuses storage — worth knowing about. Keep the value in memory so
      // the tab still behaves as if the write landed.
      Sentry.captureException(error);
      memoryFor(area)?.set(key, value);
    }
  }

  function clean(key: K): void {
    const store = nativeStorage(area);
    memoryFor(area)?.delete(key);
    if (!store) return;
    try {
      store.removeItem(key);
    } catch (error) {
      Sentry.captureException(error);
    }
  }

  return { get, set, clean };
}
