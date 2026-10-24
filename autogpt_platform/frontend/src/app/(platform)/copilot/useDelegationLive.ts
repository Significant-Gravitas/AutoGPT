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
import { useDelegationAnswerStore } from "./delegationAnswerStore";
import {
  askedQuestionOf,
  pendingQuestionOf,
  resolveLiveStatus,
} from "./delegationLiveStatus";
import type {
  ChatDelegation,
  DelegationExpert,
  LiveDelegationStatus,
} from "./delegations";
import { resolveDelegationExpert } from "./delegationViews";
import { useExpertMap } from "./useExpertMap";

const TICKING = new Set<LiveDelegationStatus>([
  "running",
  "queued",
  "needs-input",
]);

export interface LiveDelegation {
  status: LiveDelegationStatus;
  expert: DelegationExpert;
  question: string | null;
  questionOptions: string[];
  /** When the teammate asked, for "paused · 3m". */
  askedAt: number | null;
  /** What the user answered from this chat, until the teammate resumes. */
  answer: string | null;
  elapsedSeconds: number | null;
  response: string | null;
  session: SessionDetailResponse | null;
  steps: ChainRow[];
  latestText: string | null;
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

function startOf(delegation: ChatDelegation, session: SessionDetailResponse) {
  const started = Date.parse(delegation.startedAt ?? session.created_at);
  return Number.isFinite(started) ? started : null;
}

export function useDelegationLive(delegation: ChatDelegation): LiveDelegation {
  const { expertsById } = useExpertMap();
  const subSessionId = delegation.subSessionId;
  // A finished run is fetched once for its steps; the poll stops itself on
  // an idle session, so only an in-flight run keeps refetching.
  const { session, isError, isPaused } = useLiveSubSession(
    subSessionId ?? "",
    !!subSessionId,
  );
  const answer = useDelegationAnswerStore((s) =>
    subSessionId ? (s.answers[subSessionId] ?? null) : null,
  );
  const turn = session ? collectCurrentTurn(session) : null;
  const isLive = !!session && isSessionLive(session);
  const pending = session ? pendingQuestionOf(session) : null;
  const asked = turn ? askedQuestionOf(turn.steps) : null;
  const polledQuestion =
    session && !isLive ? (pending?.text ?? asked?.text ?? null) : null;
  const question =
    polledQuestion ??
    (delegation.status === "needs-input" && !session
      ? delegation.question
      : null);

  const ticking = !!answer || !!question || TICKING.has(delegation.status);
  const now = useNow(ticking);
  const status = resolveLiveStatus({
    delegation,
    session,
    isLive,
    question,
    isError,
    isPaused,
    answer,
    now,
  });
  const running = status === "running" || status === "queued";

  let elapsedSeconds = delegation.elapsedSeconds;
  const started = session ? startOf(delegation, session) : null;
  if (running && started !== null)
    elapsedSeconds = Math.max(0, (now - started) / 1000);

  const steps = (turn?.steps ?? []).map((step, i, all) =>
    toMiniRow(step, i, isLive && i === all.length - 1),
  );
  const finishedLive =
    status === "completed" && delegation.status !== "completed";

  return {
    status,
    expert: resolveDelegationExpert(delegation, expertsById),
    question: status === "needs-input" ? question : null,
    questionOptions:
      status === "needs-input"
        ? asked?.options.length
          ? asked.options
          : delegation.questionOptions
        : [],
    askedAt: pending?.askedAt ?? null,
    answer: answer && status !== "needs-input" ? answer.text : null,
    elapsedSeconds,
    response: finishedLive
      ? (turn?.latestText ?? delegation.response)
      : delegation.response,
    session,
    steps,
    latestText: turn?.latestText ?? null,
  };
}
