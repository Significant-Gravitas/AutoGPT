import { useSyncExternalStore } from "react";

/**
 * A framer-motion `initial` for chrome that is in the server-rendered HTML.
 * Server render and hydration get `false`, so the HTML holds the end state
 * rather than `opacity: 0` until the JavaScript runs; later mounts animate in.
 */
export function useEntranceInitial<T>(initial: T): T | false {
  const isHydrated = useSyncExternalStore(
    subscribeToNothing,
    () => true,
    () => false,
  );
  return isHydrated ? initial : false;
}

function subscribeToNothing() {
  return () => {};
}
