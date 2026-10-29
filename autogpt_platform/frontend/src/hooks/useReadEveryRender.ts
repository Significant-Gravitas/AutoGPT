/**
 * Calls `read` on every render and returns what it returns, like a plain call
 * in the component body. The React Compiler caches plain calls by their
 * arguments, which would freeze a value read from outside React (storage,
 * the clock, a store's `getState()`); it never caches a hook call. This hook
 * is not compiled itself, so the call below runs every time.
 */
export function useReadEveryRender<T>(read: () => T): T {
  "use no memo";
  return read();
}
