import {
  useGetV2ListChatTransports,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef } from "react";
import {
  clearKickoffPending,
  getKickoffStatus,
  markKickoffDone,
  markKickoffPending,
  withKickoffLock,
} from "./expertKickoff";
import { latestExpertSessionParams } from "./expertSessionQuery";

interface Args {
  expertId: string | null;
  kickoff: boolean;
  sessionId: string | null;
  sessionExpertId: string | null;
  hasPersistedMessages: boolean | null;
  isClientThreadEmpty: boolean;
  onAdoptSession: (sessionId: string) => void;
  onKickoff: (expertId: string) => Promise<void>;
  onSettled: () => void;
}

export function useExpertKickoff({
  expertId,
  kickoff,
  sessionId,
  sessionExpertId,
  hasPersistedMessages,
  isClientThreadEmpty,
  onAdoptSession,
  onKickoff,
  onSettled,
}: Args) {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const firedRef = useRef(false);
  const settledRef = useRef(false);
  const isArmed = isExpertsEnabled && kickoff && !!expertId;
  const wantsSession = isArmed && !sessionId && !!expertId;
  const wantsCreatePath = wantsSession && getKickoffStatus(expertId) === "idle";

  const sessionsQuery = useGetV2ListSessions(
    latestExpertSessionParams(expertId),
    {
      query: {
        enabled: wantsSession,
        refetchOnWindowFocus: false,
        refetchInterval:
          wantsSession && getKickoffStatus(expertId) === "pending"
            ? 1_000
            : false,
      },
    },
  );
  const transportsQuery = useGetV2ListChatTransports({
    query: { enabled: wantsCreatePath },
  });

  const onKickoffRef = useRef(onKickoff);
  const onAdoptSessionRef = useRef(onAdoptSession);
  const onSettledRef = useRef(onSettled);

  useEffect(() => {
    onKickoffRef.current = onKickoff;
    onAdoptSessionRef.current = onAdoptSession;
    onSettledRef.current = onSettled;
  }, [onAdoptSession, onKickoff, onSettled]);

  const settleKickoff = useRef(() => {
    if (settledRef.current) return;
    settledRef.current = true;
    onSettledRef.current();
  }).current;

  const fireKickoff = useRef(async (id: string) => {
    try {
      await onKickoffRef.current(id);
    } catch {
      clearKickoffPending(id);
    }
  }).current;

  const beginKickoff = useRef((id: string) => {
    if (firedRef.current) return;
    firedRef.current = true;
    void withKickoffLock(id, async () => {
      if (getKickoffStatus(id) !== "idle") {
        firedRef.current = false;
        return;
      }
      markKickoffPending(id);
      await fireKickoff(id);
    });
  }).current;

  useEffect(() => {
    if (!isArmed || !expertId) return;
    if (getKickoffStatus(expertId) !== "done") return;
    settleKickoff();
  }, [expertId, isArmed, settleKickoff]);

  useEffect(() => {
    if (!isArmed || !expertId || sessionId) return;
    const sessions =
      sessionsQuery.data?.status === 200
        ? sessionsQuery.data.data.sessions
        : null;
    if (sessions !== null && sessions.length > 0) {
      onAdoptSessionRef.current(sessions[0].id);
      return;
    }
    if (sessions === null && !sessionsQuery.isError) return;
    if (getKickoffStatus(expertId) !== "idle") return;
    if (transportsQuery.data?.status !== 200) return;
    beginKickoff(expertId);
  }, [
    beginKickoff,
    expertId,
    isArmed,
    sessionId,
    sessionsQuery.data,
    sessionsQuery.isError,
    transportsQuery.data,
  ]);

  useEffect(() => {
    if (!isArmed || !expertId || !sessionId) return;
    if (sessionExpertId !== expertId || hasPersistedMessages === null) return;

    if (hasPersistedMessages) {
      markKickoffDone(expertId);
      settleKickoff();
      return;
    }

    if (!isClientThreadEmpty) return;
    if (getKickoffStatus(expertId) !== "idle") return;
    beginKickoff(expertId);
  }, [
    beginKickoff,
    expertId,
    hasPersistedMessages,
    isArmed,
    isClientThreadEmpty,
    sessionExpertId,
    sessionId,
    settleKickoff,
  ]);

  return {
    isKickoffStarting:
      isExpertsEnabled &&
      kickoff &&
      !!expertId &&
      !sessionId &&
      getKickoffStatus(expertId) !== "done",
  };
}
