"use client";

import { useContext, useState } from "react";

import { parseTurnPause } from "../../turnPause";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import { CurrentPauseContext } from "./CurrentPauseContext";
import { isPauseStopped, rememberStoppedPause } from "./helpers";

interface Args {
  data: unknown;
  /** Stable for the pause across reloads: its marker row, when it has one. */
  pauseKey: string;
}

export function useTurnPausedCard({ data, pauseKey }: Args) {
  const actions = useContext(CopilotChatActionsContext);
  const current = useContext(CurrentPauseContext);
  const sessionId = actions?.sessionId ?? null;
  const storageKey = sessionId ? `${sessionId}:${pauseKey}` : null;
  const [isStopped, setIsStopped] = useState(
    () => storageKey !== null && isPauseStopped(storageKey),
  );
  const [isPickedUp, setIsPickedUp] = useState(false);
  const isTracked = current !== undefined;
  const isCurrent = data != null && data === current;
  // Acting needs a chat to act on and a way to follow the turn the server
  // starts; the share viewer and the side panels have neither.
  const canAct = Boolean(sessionId && actions?.onBackendTurn);

  function stop() {
    if (storageKey) rememberStoppedPause(storageKey);
    setIsStopped(true);
  }

  function followTurn() {
    setIsPickedUp(true);
    actions?.onBackendTurn?.();
  }

  return {
    pause: parseTurnPause(data),
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
