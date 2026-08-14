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
} from "./expertKickoff";

interface Args {
  expertId: string | null;
  kickoff: boolean;
  sessionId: string | null;
  /** Expert the OPEN session belongs to, from the session response. */
  sessionExpertId: string | null;
  /** True when the open session is empty by both the server's account and the
   *  client's; null while either is still unknown. */
  isThreadEmpty: boolean | null;
  onAdoptSession: (sessionId: string) => void;
  onKickoff: (expertId: string) => Promise<void>;
}

// Day-one kickoff: when the copilot mounts from `/copilot?expertId=<id>&kickoff=1`
// and the expert has no thread yet, auto-create the expert session and send the
// hidden kickoff message so the expert introduces itself and starts working.
//
// Layered duplicate protection, weakest to strongest:
// 1. `firedRef` — one fire per mount (StrictMode replays, re-renders).
// 2. localStorage state machine — `pending:<ts>` is set synchronously before
//    any async work, so a second tab reads it and stands down; `done` is only
//    written after the send is accepted, and `pending` is cleared on failure
//    so the standard error state remains retryable. A crashed tab's `pending`
//    expires rather than consuming the kickoff forever.
// 3. The deterministic kickoff `message_id` (see expertKickoff.ts) — the
//    Postgres PK is the atomic boundary that guarantees the turn and its
//    workflow side effects fire at most once, whatever the tabs do.
export function useExpertKickoff({
  expertId,
  kickoff,
  sessionId,
  sessionExpertId,
  isThreadEmpty,
  onAdoptSession,
  onKickoff,
}: Args) {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);
  const firedRef = useRef(false);

  const isArmed =
    isExpertsEnabled && kickoff && !!expertId && !firedRef.current;
  const wantsCreatePath =
    isArmed &&
    !sessionId &&
    !!expertId &&
    getKickoffStatus(expertId) === "idle";

  // Same query key as `useChatSession`'s thread-adoption lookup — react-query
  // dedupes them to one request.
  const sessionsQuery = useGetV2ListSessions(
    { expert_id: expertId ?? undefined, limit: 1 },
    {
      query: {
        enabled: wantsCreatePath,
        refetchOnWindowFocus: false,
      },
    },
  );

  // Creating the first session needs the LLM transport inventory loaded, or
  // `createSession` throws "AI connections are still loading". Gate the send on
  // it (same query key as `useChatSession`, so it dedupes to one request).
  const transportsQuery = useGetV2ListChatTransports({
    query: { enabled: wantsCreatePath },
  });

  // Always call the latest closures — they capture the current send path,
  // which changes as the page re-renders.
  const onKickoffRef = useRef(onKickoff);
  onKickoffRef.current = onKickoff;
  const onAdoptSessionRef = useRef(onAdoptSession);
  onAdoptSessionRef.current = onAdoptSession;
  const refetchSessionsRef = useRef(sessionsQuery.refetch);
  refetchSessionsRef.current = sessionsQuery.refetch;

  async function fire(id: string) {
    try {
      await onKickoffRef.current(id);
      markKickoffDone(id);
    } catch {
      // The failure already surfaced through the standard error paths
      // (createSession toast / stream error card). Release the latch so the
      // next visit can retry instead of landing on a silent empty thread.
      clearKickoffPending(id);
    }
  }

  // Fresh-hire path: no session yet → create one and send the kickoff.
  useEffect(() => {
    if (!isArmed || !expertId || sessionId) return;
    if (sessionsQuery.data?.status !== 200) return;
    if (sessionsQuery.data.data.sessions.length > 0) return;
    if (transportsQuery.data?.status !== 200) return;
    if (getKickoffStatus(expertId) !== "idle") return;

    firedRef.current = true;
    markKickoffPending(expertId);
    void (async () => {
      // Cross-tab belt and braces: re-check the list fresh immediately before
      // creating, and adopt any session another tab produced since the cached
      // read instead of racing it with a second create.
      try {
        const fresh = await refetchSessionsRef.current();
        const sessions =
          fresh.data?.status === 200 ? fresh.data.data.sessions : [];
        if (sessions.length > 0) {
          clearKickoffPending(expertId);
          onAdoptSessionRef.current(sessions[0].id);
          return;
        }
      } catch {
        // Proceed: the deterministic message id makes a duplicate harmless.
      }
      await fire(expertId);
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isArmed, expertId, sessionId, sessionsQuery.data, transportsQuery.data]);

  // Open-session path: the deep link adopted (or another tab created) the
  // expert's thread. A thread with history retires the kickoff for good; an
  // EMPTY expert thread means a previous kickoff created the session but died
  // before its first send flushed, so resend into it. `isThreadEmpty` is only
  // true when the client-side list is also empty, which excludes this tab's
  // own in-flight first send; concurrent tabs collapse onto the same
  // deterministic message id anyway.
  useEffect(() => {
    if (!isArmed || !expertId || !sessionId) return;
    if (sessionExpertId !== expertId) return;
    if (isThreadEmpty === null) return;

    if (!isThreadEmpty) {
      if (getKickoffStatus(expertId) !== "done") markKickoffDone(expertId);
      return;
    }

    firedRef.current = true;
    markKickoffPending(expertId);
    void fire(expertId);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isArmed, expertId, sessionId, sessionExpertId, isThreadEmpty]);
}
