"use client";

import { useEffect, useState } from "react";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import {
  collectCurrentTurn,
  isSessionLive,
  toMiniRow,
  useLiveSubSession,
} from "./components/ToolChain/SubSessionLive";
import type { ChainRow } from "./components/ToolChain/helpers";
import type {
  ChatDelegation,
  DelegationStatus,
  LiveDelegationStatus,
} from "./delegations";

export interface LiveDelegation {
  status: LiveDelegationStatus;
  question: string | null;
  elapsedSeconds: number | null;
  session: SessionDetailResponse | null;
  steps: ChainRow[];
  latestText: string | null;
}

const IN_FLIGHT = new Set<DelegationStatus>(["running", "queued"]);

function pendingQuestionOf(session: SessionDetailResponse): string | null {
  const question = session.metadata?.pending_question;
  return question && typeof question.text === "string" && question.text
    ? question.text
    : null;
}

/** Ticks once a second while `active`, so a running hand-off's elapsed time
 *  keeps moving between polls. */
function useNow(active: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [active]);
  return now;
}

export function useDelegationLive(delegation: ChatDelegation): LiveDelegation {
  const frozenInFlight = IN_FLIGHT.has(delegation.status);
  const shouldPoll = frozenInFlight && !!delegation.subSessionId;
  const { session, isError, isPaused } = useLiveSubSession(
    delegation.subSessionId ?? "",
    shouldPoll,
  );
  const live = !!session && isSessionLive(session);
  const question = session ? pendingQuestionOf(session) : null;

  let status: LiveDelegationStatus = delegation.status;
  if (shouldPoll) {
    if (isError || isPaused) status = "unknown";
    else if (session && !live) status = question ? "needs-input" : "completed";
  }

  const running = status === "running" || status === "queued";
  const now = useNow(running);
  let elapsedSeconds = delegation.elapsedSeconds;
  if (running && session) {
    const startedAt = Date.parse(session.created_at);
    if (Number.isFinite(startedAt))
      elapsedSeconds = Math.max(0, (now - startedAt) / 1000);
  }

  const turn = session ? collectCurrentTurn(session) : null;
  const steps = (turn?.steps ?? []).map((step, i, all) =>
    toMiniRow(step, i, live && i === all.length - 1),
  );

  return {
    status,
    question: status === "needs-input" ? question : null,
    elapsedSeconds,
    session,
    steps,
    latestText: turn?.latestText ?? null,
  };
}
