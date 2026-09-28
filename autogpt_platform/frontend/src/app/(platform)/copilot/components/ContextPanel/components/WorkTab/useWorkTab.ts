"use client";

import { useState } from "react";
import { useChatSessionDelegations } from "../../useChatSessionDelegations";

export function useWorkTab(sessionId: string | null) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { delegations, liveStatuses, reportStatus, rearm, isLoading, isError } =
    useChatSessionDelegations(sessionId);
  const selected = delegations.find((d) => d.toolCallId === selectedId) ?? null;

  function select(toolCallId: string | null) {
    rearm();
    setSelectedId(toolCallId);
  }

  return {
    delegations,
    liveStatuses,
    reportStatus,
    rearm,
    selected,
    select,
    isLoading,
    isError,
  };
}
