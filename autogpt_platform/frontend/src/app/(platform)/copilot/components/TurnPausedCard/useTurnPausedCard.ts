"use client";

import { useContext, useState } from "react";

import { parseTurnPause } from "../../turnPause";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { CurrentPauseContext } from "./CurrentPauseContext";
import { useStopHere } from "./useStopHere";

interface Args {
  data: unknown;
  /** Stable for the pause across reloads when it is its stored row's. */
  pauseKey: string;
  /** The pause is a stored row, so its key outlives a reload. */
  isStoredRow: boolean;
}

export function useTurnPausedCard({ data, pauseKey, isStoredRow }: Args) {
  const actions = useContext(CopilotChatActionsContext);
  const current = useContext(CurrentPauseContext);
  const pause = parseTurnPause(data);
  const sessionId = actions?.sessionId ?? null;
  const [isPickedUp, setIsPickedUp] = useState(false);
  const isTracked = current !== undefined;
  const isCurrent = data != null && data === current;
  const { isStopped, stop } = useStopHere({
    sessionId,
    pause,
    pauseKey,
    isStoredRow,
    isCurrent,
  });
  // Acting needs a chat to act on and a way to follow the turn the server
  // starts; the share viewer and the side panels have neither.
  const canAct = Boolean(sessionId && actions?.onBackendTurn);

  function followTurn() {
    setIsPickedUp(true);
    actions?.onBackendTurn?.();
  }

  return {
    pause,
    sessionId: canAct ? sessionId : null,
    assistantName: actions?.assistantName ?? "Otto",
    isLive: isCurrent && canAct && !isStopped && !isPickedUp,
    note: isStopped
      ? "Stopped here"
      : isPickedUp || (isTracked && !isCurrent)
        ? "Picked back up"
        : null,
    stop,
    followTurn,
  };
}
