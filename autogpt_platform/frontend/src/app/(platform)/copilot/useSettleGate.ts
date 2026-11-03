import { useEffect, useInsertionEffect, useRef } from "react";

/**
 * Lets a send wait until the chat's current turn has settled. Each waiter
 * learns whether it settled (send now) or the hook went away first (the
 * chat host unmounted), in which case the send is dropped.
 */
export function useSettleGate(isSettled: boolean) {
  const isSettledRef = useRef(isSettled);
  // Every commit, before the effects below resolve any waiter.
  useInsertionEffect(() => {
    isSettledRef.current = isSettled;
  });
  const waitersRef = useRef<Array<(settled: boolean) => void>>([]);
  const isMountedRef = useRef(true);

  useEffect(() => {
    if (!isSettled) return;
    waitersRef.current.splice(0).forEach((resolve) => resolve(true));
  }, [isSettled]);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      waitersRef.current.splice(0).forEach((resolve) => resolve(false));
    };
  }, []);

  /** A send still uploading when the host unmounted asks only afterwards,
   *  when nothing is left to resolve it. */
  function waitUntilSettled() {
    if (!isMountedRef.current) return Promise.resolve(false);
    if (isSettledRef.current) return Promise.resolve(true);
    return new Promise<boolean>((resolve) => {
      waitersRef.current.push(resolve);
    });
  }

  /** A turn was just dispatched: waiters that arrive before the next render
   *  must queue behind it rather than see the stale settled state. */
  function markUnsettled() {
    isSettledRef.current = false;
  }

  return { waitUntilSettled, markUnsettled };
}
