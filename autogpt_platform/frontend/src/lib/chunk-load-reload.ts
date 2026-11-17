import { isChunkLoadFailure } from "@/lib/sentry-filters";
import { capturePostHogEvent } from "@/services/analytics/posthog-capture";
import { DeploySkewEvent } from "@/services/analytics/posthog-events";

const CHUNK_RELOAD_GUARD_KEY = "chunk_reload_at";
const CHUNK_RELOAD_COOLDOWN_MS = 60_000;

/**
 * True when the browser could not fetch code the page loads on demand. After
 * a deploy, a tab still running the old build asks for chunks the new
 * deployment no longer serves, and every later click that needs one fails
 * the same way until the page is reloaded.
 */
export function isChunkLoadError(error: unknown): boolean {
  return (
    error instanceof Error && isChunkLoadFailure(error.name, error.message)
  );
}

/**
 * Reloads the page once so the tab picks up the current build. A timestamp in
 * sessionStorage stops a chunk that fails for any other reason (network,
 * blocked script) from trapping the tab in a reload loop. Returns whether the
 * page is reloading.
 */
export function reloadForChunkLoadError(
  error: unknown,
  surface: "unhandled_rejection" | "error_boundary",
): boolean {
  if (!isChunkLoadError(error)) return false;

  const lastReloadAt = Number(readReloadGuard());
  if (Date.now() - lastReloadAt < CHUNK_RELOAD_COOLDOWN_MS) return false;
  if (!writeReloadGuard(Date.now())) return false;

  capturePostHogEvent(DeploySkewEvent.CHUNK_RELOAD_TRIGGERED, {
    surface,
    path: window.location.pathname,
  });
  window.location.reload();
  return true;
}

/**
 * A chunk that a click handler imports fails as an unhandled promise
 * rejection, which no React error boundary sees: the click just does nothing.
 */
export function reloadOnUnhandledChunkLoadError() {
  function onUnhandledRejection(event: PromiseRejectionEvent) {
    reloadForChunkLoadError(event.reason, "unhandled_rejection");
  }
  window.addEventListener("unhandledrejection", onUnhandledRejection);
  return () =>
    window.removeEventListener("unhandledrejection", onUnhandledRejection);
}

// The guard goes straight to the browser's sessionStorage, not the safe
// wrapper: the wrapper falls back to memory, which the reload itself wipes.
function readReloadGuard(): string | null {
  try {
    return window.sessionStorage.getItem(CHUNK_RELOAD_GUARD_KEY);
  } catch {
    return null;
  }
}

function writeReloadGuard(at: number): boolean {
  try {
    window.sessionStorage.setItem(CHUNK_RELOAD_GUARD_KEY, String(at));
    return window.sessionStorage.getItem(CHUNK_RELOAD_GUARD_KEY) === String(at);
  } catch {
    return false;
  }
}
