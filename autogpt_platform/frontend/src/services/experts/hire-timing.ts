// How long a hire actually took, measured from the click that opened the flow
// to the moment it finishes. sessionStorage rather than component state
// because the flow can start on the marketplace profile sheet and finish
// inside the copilot dialog, and the two never share a React tree.

const KEY_PREFIX = "autogpt:hire-started:";

function storage() {
  if (typeof window === "undefined") return null;
  try {
    return window.sessionStorage;
  } catch {
    return null;
  }
}

export function markHireStarted(templateId: string) {
  try {
    storage()?.setItem(`${KEY_PREFIX}${templateId}`, String(Date.now()));
  } catch {
    // Private-mode and in-app browsers can refuse writes; a missing timing
    // is never worth breaking a hire over.
  }
}

/** Reads the mark and clears it, so a second finish cannot report a stale span. */
export function takeHireElapsedMs(templateId: string): number | null {
  const store = storage();
  if (!store) return null;
  const key = `${KEY_PREFIX}${templateId}`;
  try {
    const raw = store.getItem(key);
    store.removeItem(key);
    if (!raw) return null;
    const startedAt = Number(raw);
    if (!Number.isFinite(startedAt)) return null;
    return Math.max(0, Date.now() - startedAt);
  } catch {
    return null;
  }
}
