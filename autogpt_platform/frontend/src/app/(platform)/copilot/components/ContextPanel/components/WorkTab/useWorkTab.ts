"use client";

import { useState } from "react";
import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import { convertChatSessionMessagesToUiMessages } from "../../../../helpers/convertChatSessionToUiMessages";
import {
  type ChatDelegation,
  type LiveDelegationStatus,
  getChatDelegations,
} from "../../../../delegations";

const POLL_MS = 5000;

/** The chat's hand-offs, read off its own persisted transcript so the panel
 *  needs nothing from the chat column. Shares the chat's session query and
 *  keeps it fresh while any hand-off is still in flight. */
export function useWorkTab(sessionId: string | null) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [liveStatuses, setLiveStatuses] = useState<
    Record<string, LiveDelegationStatus>
  >({});
  const { data, isLoading, isError } = useGetV2GetSession(
    sessionId ?? "",
    undefined,
    {
      query: {
        enabled: !!sessionId,
        refetchInterval: (query) => {
          const raw = query.state.data;
          const session = raw && raw.status === 200 ? raw.data : null;
          if (!session) return false;
          const inFlight = getChatDelegations(
            convertChatSessionMessagesToUiMessages(
              session.id,
              session.messages ?? [],
              { isComplete: true },
            ).messages,
          ).some((d) => d.status === "running" || d.status === "queued");
          return inFlight || !!session.active_stream ? POLL_MS : false;
        },
      },
    },
  );
  const session = data && data.status === 200 ? data.data : null;
  const delegations: ChatDelegation[] = session
    ? getChatDelegations(
        convertChatSessionMessagesToUiMessages(
          session.id,
          session.messages ?? [],
          { isComplete: true },
        ).messages,
      )
    : [];
  const selected = delegations.find((d) => d.toolCallId === selectedId) ?? null;

  function reportStatus(toolCallId: string, status: LiveDelegationStatus) {
    setLiveStatuses((prev) =>
      prev[toolCallId] === status ? prev : { ...prev, [toolCallId]: status },
    );
  }

  return {
    delegations,
    liveStatuses,
    reportStatus,
    selected,
    select: setSelectedId,
    isLoading: isLoading && !session,
    isError,
  };
}
