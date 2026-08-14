import {
  useGetV2ListChatTransports,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef, useState } from "react";
import {
  clearKickoffPending,
  getKickoffStatus,
  markKickoffDone,
  markKickoffPending,
  type KickoffAttemptToken,
  withKickoffLock,
} from "./expertKickoff";
import { latestExpertSessionParams } from "./expertSessionQuery";

interface Args {
  userId: string | null;
  expertId: string | null;
  kickoff: boolean;
  sessionId: string | null;
  sessionExpertId: string | null;
  hasPersistedExpertHistory: boolean | null;
  kickoffAttemptToken: KickoffAttemptToken | null;
  isClientThreadEmpty: boolean;
  onAdoptSession: (sessionId: string) => void;
  onKickoff: (
    expertId: string,
    attemptToken: KickoffAttemptToken,
  ) => Promise<void>;
  onSettled: () => void;
}

export function useExpertKickoff({
  userId,
  expertId,
  kickoff,
  sessionId,
  sessionExpertId,
  hasPersistedExpertHistory,
  kickoffAttemptToken,
  isClientThreadEmpty,
  onAdoptSession,
  onKickoff,
  onSettled,
}: Args) {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const firedRef = useRef(false);
  const settledRef = useRef(false);
  const attemptTokenRef = useRef<KickoffAttemptToken | null>(null);
  const [failedKickoff, setFailedKickoff] = useState<{
    userId: string;
    expertId: string;
  } | null>(null);
  const isArmed = isExpertsEnabled && kickoff && !!userId && !!expertId;
  const wantsSession = isArmed && !sessionId && !!expertId;
  const wantsCreatePath =
    wantsSession &&
    !!userId &&
    !!expertId &&
    getKickoffStatus(userId, expertId) === "idle";

  const sessionsQuery = useGetV2ListSessions(
    latestExpertSessionParams(expertId),
    {
      query: {
        enabled: wantsSession,
        refetchOnWindowFocus: false,
        refetchInterval:
          wantsSession &&
          userId &&
          expertId &&
          getKickoffStatus(userId, expertId) === "pending"
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

  useEffect(() => {
    firedRef.current = false;
    settledRef.current = false;
    attemptTokenRef.current = null;
  }, [expertId, userId]);

  useEffect(() => {
    if (kickoffAttemptToken) attemptTokenRef.current = kickoffAttemptToken;
  }, [kickoffAttemptToken]);

  const settleKickoff = useRef(() => {
    if (settledRef.current) return;
    settledRef.current = true;
    onSettledRef.current();
  }).current;

  const fireKickoff = useRef(
    async (ownerId: string, id: string, attemptToken: KickoffAttemptToken) => {
      try {
        await onKickoffRef.current(id, attemptToken);
      } catch {
        clearKickoffPending(ownerId, id, attemptToken);
        firedRef.current = false;
        setFailedKickoff({ userId: ownerId, expertId: id });
        settleKickoff();
      }
    },
  ).current;

  const beginKickoff = useRef((ownerId: string, id: string) => {
    if (firedRef.current) return;
    firedRef.current = true;
    void withKickoffLock(ownerId, id, async () => {
      if (getKickoffStatus(ownerId, id) !== "idle") {
        firedRef.current = false;
        return;
      }
      const attemptToken = markKickoffPending(ownerId, id);
      attemptTokenRef.current = attemptToken;
      await fireKickoff(ownerId, id, attemptToken);
    }).catch(() => {
      firedRef.current = false;
      setFailedKickoff({ userId: ownerId, expertId: id });
      settleKickoff();
    });
  }).current;

  useEffect(() => {
    if (!isArmed || !userId || !expertId) return;
    if (getKickoffStatus(userId, expertId) !== "done") return;
    settleKickoff();
  }, [expertId, isArmed, settleKickoff, userId]);

  useEffect(() => {
    if (!isArmed || !userId || !expertId || sessionId) return;
    const sessions =
      sessionsQuery.data?.status === 200
        ? sessionsQuery.data.data.sessions
        : null;
    if (sessions !== null && sessions.length > 0) {
      onAdoptSessionRef.current(sessions[0].id);
      return;
    }
    if (sessions === null && !sessionsQuery.isError) return;
    if (getKickoffStatus(userId, expertId) !== "idle") return;
    if (transportsQuery.data?.status !== 200) return;
    beginKickoff(userId, expertId);
  }, [
    beginKickoff,
    expertId,
    isArmed,
    sessionId,
    sessionsQuery.data,
    sessionsQuery.isError,
    transportsQuery.data,
    userId,
  ]);

  useEffect(() => {
    if (!isArmed || !userId || !expertId || !sessionId) return;
    if (sessionExpertId !== expertId || hasPersistedExpertHistory === null)
      return;

    if (hasPersistedExpertHistory) {
      const attemptToken = attemptTokenRef.current;
      if (attemptToken) {
        void withKickoffLock(userId, expertId, async () => {
          markKickoffDone(userId, expertId, attemptToken);
        }).catch(() => undefined);
      } else {
        void withKickoffLock(userId, expertId, async () => {
          if (getKickoffStatus(userId, expertId) !== "idle") return;
          const historyAttempt = markKickoffPending(userId, expertId);
          markKickoffDone(userId, expertId, historyAttempt);
        }).catch(() => undefined);
      }
      settleKickoff();
      return;
    }

    if (!isClientThreadEmpty) return;
    if (getKickoffStatus(userId, expertId) !== "idle") return;
    beginKickoff(userId, expertId);
  }, [
    beginKickoff,
    expertId,
    hasPersistedExpertHistory,
    isArmed,
    isClientThreadEmpty,
    sessionExpertId,
    sessionId,
    settleKickoff,
    userId,
  ]);

  return {
    isKickoffStarting:
      isExpertsEnabled &&
      kickoff &&
      !!userId &&
      !!expertId &&
      !(
        failedKickoff?.userId === userId && failedKickoff.expertId === expertId
      ) &&
      getKickoffStatus(userId, expertId) !== "done",
  };
}
