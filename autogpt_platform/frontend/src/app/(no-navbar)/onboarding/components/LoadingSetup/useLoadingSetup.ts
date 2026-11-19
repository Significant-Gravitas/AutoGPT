import { useEffect, useRef, useState } from "react";

// The wizard's other reads each give up on their own; the signed-in account
// has no deadline, and it is what #16965 stalled on.
export const SETUP_LOAD_DEADLINE_MS = 15_000;

interface Args {
  waitingFor: string[];
  onSlow: () => void;
}

export function useLoadingSetup({ waitingFor, onSlow }: Args) {
  const [timedOut, setTimedOut] = useState(false);
  const noticeRef = useRef<HTMLDivElement>(null);
  const latestOnSlow = useRef(onSlow);
  const isWaitingForAccount = waitingFor.includes("account");

  useEffect(() => {
    latestOnSlow.current = onSlow;
  });

  useEffect(() => {
    if (!isWaitingForAccount) return;
    const timer = setTimeout(() => {
      setTimedOut(true);
      latestOnSlow.current();
    }, SETUP_LOAD_DEADLINE_MS);
    return () => clearTimeout(timer);
  }, [isWaitingForAccount]);

  const isSlow = timedOut && isWaitingForAccount;

  // The notice replaces the placeholder, so focus moves with it.
  useEffect(() => {
    if (isSlow) noticeRef.current?.focus();
  }, [isSlow]);

  function reload() {
    window.location.reload();
  }

  return { isSlow, noticeRef, reload };
}
