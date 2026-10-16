import { useEffect, useRef } from "react";

/**
 * Lets a send wait until the chat's current turn has settled. Each waiter
 * learns whether it settled (send now) or the hook went away first (the
 * chat host unmounted), in which case the send is dropped.
 */
export function useSettleGate(isSettled: boolean) {
  const isSettledRef = useRef(isSettled);
  isSettledRef.current = isSettled;
  const waitersRef = useRef<Array<(settled: boolean) => void>>([]);

  useEffect(() => {
    if (!isSettled) return;
    waitersRef.current.splice(0).forEach((resolve) => resolve(true));
  }, [isSettled]);
  useEffect(
    () => () => {
      waitersRef.current.splice(0).forEach((resolve) => resolve(false));
    },
    [],
  );

  function waitUntilSettled() {
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
