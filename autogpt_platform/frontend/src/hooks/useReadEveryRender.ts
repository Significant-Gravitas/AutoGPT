import { useSyncExternalStore } from "react";

function subscribeToNothing() {
  return () => {};
}

/**
 * Calls `read` on every render, like a plain call in the component body.
 * The React Compiler caches plain calls by their arguments, which would
 * freeze a value read from outside React (storage, the clock, a store's
 * `getState()`); a hook call is never cached. It does not subscribe, so a
 * change still shows on the next render, exactly as before.
 */
export function useReadEveryRender<T>(read: () => T): T {
  return useSyncExternalStore(subscribeToNothing, read, read);
}
