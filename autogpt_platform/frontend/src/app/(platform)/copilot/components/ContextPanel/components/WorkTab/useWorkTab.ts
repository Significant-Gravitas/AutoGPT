"use client";

import { useState } from "react";
import {
  useChatSessionDelegations,
  useLiveStatuses,
} from "../../useChatSessionDelegations";

export function useWorkTab(sessionId: string | null) {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const { delegations, isLoading, isError } =
    useChatSessionDelegations(sessionId);
  const { liveStatuses, reportStatus } = useLiveStatuses();
  const selected = delegations.find((d) => d.toolCallId === selectedId) ?? null;

  return {
    delegations,
    liveStatuses,
    reportStatus,
    selected,
    select: setSelectedId,
    isLoading,
    isError,
  };
}
