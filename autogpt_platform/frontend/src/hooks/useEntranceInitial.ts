import { useSyncExternalStore } from "react";

/**
 * `false` while hydrating, so the server's HTML holds the end state rather
 * than `opacity: 0`; later mounts animate in.
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
