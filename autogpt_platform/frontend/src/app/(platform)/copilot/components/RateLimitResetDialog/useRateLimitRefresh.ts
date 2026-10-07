import { useEffect, useRef, useState } from "react";

function boundedRefresh(retry: (throwOnError?: boolean) => Promise<unknown>) {
  let cancel = () => {};
  const deadline = new Promise<never>((_, reject) => {
    cancel = () => reject(new Error("Usage refresh did not finish"));
  });
  const timeout = setTimeout(cancel, 25_000);
  return {
    cancel() {
      clearTimeout(timeout);
      cancel();
    },
    result: Promise.race([
      new Promise((resolve) => resolve(retry(true))),
      deadline,
    ]).finally(() => clearTimeout(timeout)),
  };
}

export function useRateLimitRefresh(
  message: string | null,
  retry: (throwOnError?: boolean) => Promise<unknown>,
  onRefreshingChange?: (refreshing: boolean) => void,
) {
  const latest = useRef({ retry, onRefreshingChange });
  latest.current = { retry, onRefreshingChange };
  const requestID = useRef(0);
  const pending = useRef<ReturnType<typeof boundedRefresh> | null>(null);
  const [checkedMessage, setCheckedMessage] = useState<string | null>(null);
  const [checking, setChecking] = useState(false);
  const [failed, setFailed] = useState(false);
  const [guarded, setGuarded] = useState(false);
  async function refresh() {
    const id = ++requestID.current;
    pending.current?.cancel();
    const request = boundedRefresh(latest.current.retry);
    pending.current = request;
    setChecking(true);
    setFailed(false);
    setGuarded(true);
    latest.current.onRefreshingChange?.(true);
    try {
      await request.result;
      if (requestID.current === id) setGuarded(false);
    } catch {
      if (requestID.current === id) setFailed(true);
    } finally {
      if (requestID.current === id) {
        pending.current = null;
        setChecking(false);
        setCheckedMessage(message);
        latest.current.onRefreshingChange?.(false);
      }
    }
  }
  function release() {
    requestID.current += 1;
    pending.current?.cancel();
    pending.current = null;
    setGuarded(false);
    setChecking(false);
    setFailed(false);
    setCheckedMessage(message);
    latest.current.onRefreshingChange?.(false);
  }
  const refreshRef = useRef(refresh);
  refreshRef.current = refresh;
  useEffect(() => {
    if (message) void refreshRef.current();
    else setCheckedMessage(null);
  }, [message]);
  useEffect(
    () => () => {
      requestID.current += 1;
      pending.current?.cancel();
    },
    [],
  );
  return {
    checking: checking || (!!message && checkedMessage !== message),
    failed,
    guarded,
    refresh,
    release,
  };
}
