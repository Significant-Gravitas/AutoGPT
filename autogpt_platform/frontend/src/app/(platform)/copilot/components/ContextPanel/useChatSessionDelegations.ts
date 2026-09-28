"use client";

import { useState } from "react";
import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { LiveDelegationStatus } from "../../delegations";
import { delegationsOf, WORK_POLL_CAP_MS, workPollInterval } from "./workPoll";

/** The chat's hand-offs, read off its own persisted transcript so the panel
 *  needs nothing from the chat column. Shares the chat's session query and
 *  keeps it fresh while a teammate is live, as its probes report it. */
export function useChatSessionDelegations(sessionId: string | null) {
  const [liveStatuses, setLiveStatuses] = useState<
    Record<string, LiveDelegationStatus>
  >({});
  const [armedAt, setArmedAt] = useState(() => Date.now());
  const { data, isLoading, isError, refetch } = useGetV2GetSession(
    sessionId ?? "",
    undefined,
    {
      query: {
        enabled: !!sessionId,
        refetchInterval: (query) => {
          const raw = query.state.data;
          return workPollInterval({
            session: raw && raw.status === 200 ? raw.data : null,
            liveStatuses,
            armedAt,
            now: Date.now(),
          });
        },
      },
    },
  );
  const session = data && data.status === 200 ? data.data : null;

  function reportStatus(toolCallId: string, status: LiveDelegationStatus) {
    setLiveStatuses((prev) =>
      prev[toolCallId] === status ? prev : { ...prev, [toolCallId]: status },
    );
  }

  /** The user is looking again: restart a capped poll. */
  function rearm() {
    const now = Date.now();
    setArmedAt(now);
    // A capped poll has stopped; one fetch lets the interval decide again.
    if (sessionId && now - armedAt > WORK_POLL_CAP_MS) void refetch();
  }

  return {
    delegations: session ? delegationsOf(session) : [],
    liveStatuses,
    reportStatus,
    rearm,
    isLoading: isLoading && !session,
    isError,
  };
}
