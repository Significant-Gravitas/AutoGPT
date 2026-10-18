"use client";

import { useEffect, useState } from "react";

import type { TurnPause } from "../../turnPause";
import {
  dropPendingStop,
  holdPendingStop,
  isPauseStopped,
  isPendingStop,
  rememberStoppedPause,
  settlePendingStop,
} from "./stoppedPauses";

interface Args {
  sessionId: string | null;
  pause: TurnPause | null;
  pauseKey: string;
  /** The key is the pause's stored row's, so it outlives a reload. */
  isStoredRow: boolean;
  isCurrent: boolean;
}

/**
 * "Stop here", kept on this device under the pause's stored row. A pause
 * still streaming in has no row yet, so a stop made then is held for the
 * session and handed to the row when it replaces the live part.
 */
export function useStopHere({
  sessionId,
  pause,
  pauseKey,
  isStoredRow,
  isCurrent,
}: Args) {
  const storageKey = sessionId ? `${sessionId}:${pauseKey}` : null;
  const hasPendingStop = Boolean(
    sessionId && pause && isCurrent && isPendingStop(sessionId, pause),
  );
  const [isStopped, setIsStopped] = useState(
    () => (storageKey !== null && isPauseStopped(storageKey)) || hasPendingStop,
  );
  const holdsPendingStop = isStopped && !isStoredRow;

  useEffect(() => {
    if (hasPendingStop && isStoredRow && sessionId && storageKey) {
      settlePendingStop(sessionId, storageKey);
    }
  }, [hasPendingStop, isStoredRow, sessionId, storageKey]);

  // The chat moved past the live pause before its row arrived: the stop
  // belongs to no later pause, however alike.
  useEffect(() => {
    if (holdsPendingStop && !isCurrent && sessionId) dropPendingStop(sessionId);
  }, [holdsPendingStop, isCurrent, sessionId]);

  function stop() {
    if (isStoredRow && storageKey) rememberStoppedPause(storageKey);
    else if (sessionId && pause) holdPendingStop(sessionId, pause);
    setIsStopped(true);
  }

  return { isStopped, stop };
}
