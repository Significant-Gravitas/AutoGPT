import { useEffect, useRef } from "react";

interface Args {
  sessionId: string | null;
  isSettled: boolean;
}

interface Waiter {
  sessionId: string;
  resolve: (isCurrent: boolean) => void;
}

/**
 * Lets a send wait until this tab has stopped drawing the current turn.
 * `status` flips to "ready" only when the smoothed stream has drained, so
 * the finish probe and any reconnect still have to settle before a new turn
 * can start without landing a user row inside the live answer.
 *
 * The promise resolves `true` once settled and `false` when the chat is
 * switched or torn down first, so a delayed send never goes to another chat.
 */
export function useLocalStreamSettle({ sessionId, isSettled }: Args) {
  const isSettledRef = useRef(isSettled);
  isSettledRef.current = isSettled;
  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;
  const waitersRef = useRef<Waiter[]>([]);
  // A 409 can arrive after the chat was torn down; with the chat gone the
  // waiter must resolve false instead of sending or hanging.
  const isMountedRef = useRef(false);

  useEffect(() => {
    if (!isSettled) return;
    const { current: waiters } = waitersRef;
    const settled = waiters.filter((waiter) => waiter.sessionId === sessionId);
    waitersRef.current = waiters.filter(
      (waiter) => waiter.sessionId !== sessionId,
    );
    settled.forEach((waiter) => waiter.resolve(true));
  }, [isSettled, sessionId]);

  useEffect(() => {
    const stale = waitersRef.current.filter(
      (waiter) => waiter.sessionId !== sessionId,
    );
    waitersRef.current = waitersRef.current.filter(
      (waiter) => waiter.sessionId === sessionId,
    );
    stale.forEach((waiter) => waiter.resolve(false));
  }, [sessionId]);

  useEffect(() => {
    isMountedRef.current = true;
    return () => {
      isMountedRef.current = false;
      waitersRef.current.splice(0).forEach((waiter) => waiter.resolve(false));
    };
  }, []);

  function waitForLocalSettle(forSessionId: string): Promise<boolean> {
    if (!isMountedRef.current || forSessionId !== sessionIdRef.current) {
      return Promise.resolve(false);
    }
    if (isSettledRef.current) return Promise.resolve(true);
    return new Promise<boolean>((resolve) => {
      waitersRef.current.push({ sessionId: forSessionId, resolve });
    });
  }

  return { waitForLocalSettle };
}
