import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { useEffect, useState } from "react";
import { POLL_CAP_MS, POLL_MS, isSessionLive } from "./helpers";

/** The polled sub-session plus the two states that used to render as
 *  nothing at all: a failed fetch, and the poll cap expiring while the run
 *  is still live. Both stop the polling, so both have to be visible —
 *  otherwise a dead card is indistinguishable from a working one. */
export function useLiveSubSession(subSessionId: string, active: boolean) {
  const [isCapped, setIsCapped] = useState(false);
  useEffect(
    function stopPollingAfterCap() {
      if (!active) return;
      const timer = setTimeout(() => setIsCapped(true), POLL_CAP_MS);
      return () => clearTimeout(timer);
    },
    [active],
  );
  const { data, isError } = useGetV2GetSession(subSessionId, undefined, {
    query: {
      enabled: active && !!subSessionId,
      refetchInterval: (query) => {
        if (query.state.status === "error" || isCapped) return false;
        const raw = query.state.data;
        const polled = raw && raw.status === 200 ? raw.data : null;
        return !polled || isSessionLive(polled) ? POLL_MS : false;
      },
      select: (res) => (res.status === 200 ? res.data : null),
    },
  });
  const session = data ?? null;
  return {
    session,
    isError,
    // A capped poll on a finished session is not "paused" — the last
    // snapshot IS the final answer, so there is nothing left to watch.
    isPaused: isCapped && (!session || isSessionLive(session)),
  };
}
