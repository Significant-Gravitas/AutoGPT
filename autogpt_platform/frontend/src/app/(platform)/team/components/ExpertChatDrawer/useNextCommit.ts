import { useEffect, useRef, useState } from "react";

/**
 * `nextCommit()` resolves once React has committed every state update queued
 * before the call: `true` when it has, `false` when the host unmounted first.
 *
 * The chat SDK flips `status` on React's sync lane, while the stream's own
 * flags (a scheduled reconnect, the post-finish probe) are plain state updates
 * that render one pass later. A check made on the `status` render alone sees
 * a turn as over while the stream still refuses to send.
 */
export function useNextCommit() {
  const [committed, setCommitted] = useState(0);
  const requestedRef = useRef(0);
  const waitersRef = useRef<Array<{ ticket: number; resolve: Resolve }>>([]);
  const isMountedRef = useRef(true);

  useEffect(() => {
    waitersRef.current = waitersRef.current.filter((waiter) => {
      if (waiter.ticket > committed) return true;
      waiter.resolve(true);
      return false;
    });
  }, [committed]);
  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      waitersRef.current.splice(0).forEach((waiter) => waiter.resolve(false));
    };
  }, []);

  function nextCommit() {
    if (!isMountedRef.current) return Promise.resolve(false);
    requestedRef.current += 1;
    const ticket = requestedRef.current;
    return new Promise<boolean>((resolve) => {
      waitersRef.current.push({ ticket, resolve });
      setCommitted((count) => Math.max(count, ticket));
    });
  }

  return nextCommit;
}

type Resolve = (committed: boolean) => void;
