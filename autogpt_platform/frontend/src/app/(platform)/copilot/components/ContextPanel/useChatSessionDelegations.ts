"use client";

import { useState } from "react";
import { useGetV2GetSession } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { convertChatSessionMessagesToUiMessages } from "../../helpers/convertChatSessionToUiMessages";
import {
  type ChatDelegation,
  type LiveDelegationStatus,
  getChatDelegations,
} from "../../delegations";

const POLL_MS = 5000;

function delegationsOf(session: SessionDetailResponse): ChatDelegation[] {
  return getChatDelegations(
    convertChatSessionMessagesToUiMessages(
      session.id,
      session.messages ?? [],
      // A turn still streaming has calls with no result yet; marking them
      // complete would read a hand-off in progress as stopped.
      { isComplete: !session.active_stream },
    ).messages,
  );
}

/** The chat's hand-offs, read off its own persisted transcript so the panel
 *  needs nothing from the chat column. Shares the chat's session query and
 *  keeps it fresh while any hand-off is still in flight. */
export function useChatSessionDelegations(sessionId: string | null) {
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
          const inFlight = delegationsOf(session).some(
            (d) => d.status === "running" || d.status === "queued",
          );
          return inFlight || !!session.active_stream ? POLL_MS : false;
        },
      },
    },
  );
  const session = data && data.status === 200 ? data.data : null;
  return {
    delegations: session ? delegationsOf(session) : [],
    isLoading: isLoading && !session,
    isError,
  };
}

/** Each hand-off's live status as its probe reports it. */
export function useLiveStatuses() {
  const [liveStatuses, setLiveStatuses] = useState<
    Record<string, LiveDelegationStatus>
  >({});
  function reportStatus(toolCallId: string, status: LiveDelegationStatus) {
    setLiveStatuses((prev) =>
      prev[toolCallId] === status ? prev : { ...prev, [toolCallId]: status },
    );
  }
  return { liveStatuses, reportStatus };
}
