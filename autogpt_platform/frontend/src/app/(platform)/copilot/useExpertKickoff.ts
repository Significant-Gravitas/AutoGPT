import {
  useGetV2ListChatTransports,
  useGetV2ListSessions,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef } from "react";
import { hasKickedOff, markKickedOff } from "./expertKickoff";

interface Args {
  expertId: string | null;
  kickoff: boolean;
  sessionId: string | null;
  onKickoff: () => void;
}

// Day-one kickoff: when the copilot mounts from `/copilot?expertId=<id>&kickoff=1`
// and the expert has no thread yet, auto-send the hidden kickoff message once so
// the expert introduces itself and starts working instead of sitting idle.
//
// The empty-sessions check reuses the same `expert_id` list query that
// `useChatSession` runs to adopt an existing thread — react-query dedupes them
// by key, so an expert that already has a thread simply gets adopted there and
// never kicks off here.
export function useExpertKickoff({
  expertId,
  kickoff,
  sessionId,
  onKickoff,
}: Args) {
  const isExpertsEnabled = useGetFlag(Flag.HIRE_EXPERTS);

  // Latched for the life of this mount so a re-render or StrictMode double
  // effect can't fire the kickoff twice; the localStorage key guards future
  // visits.
  const firedRef = useRef(false);

  const shouldKickoff =
    isExpertsEnabled &&
    kickoff &&
    !!expertId &&
    !sessionId &&
    !firedRef.current &&
    !hasKickedOff(expertId);

  const sessionsQuery = useGetV2ListSessions(
    { expert_id: expertId ?? undefined, limit: 1 },
    {
      query: {
        enabled: shouldKickoff,
        refetchOnWindowFocus: false,
      },
    },
  );

  // Creating the first session needs the LLM transport inventory loaded, or
  // `createSession` throws "AI connections are still loading". Gate the send on
  // it (same query key as `useChatSession`, so it dedupes to one request).
  const transportsQuery = useGetV2ListChatTransports({
    query: { enabled: shouldKickoff },
  });

  // Always call the latest closure — `onKickoff` captures the current send
  // path, which changes as the page re-renders.
  const onKickoffRef = useRef(onKickoff);
  onKickoffRef.current = onKickoff;

  useEffect(() => {
    if (!shouldKickoff || !expertId) return;
    if (sessionsQuery.data?.status !== 200) return;

    // The expert already has a thread, so `useChatSession` is adopting it —
    // latch and step aside without sending anything.
    if (sessionsQuery.data.data.sessions.length > 0) {
      firedRef.current = true;
      markKickedOff(expertId);
      return;
    }

    // No thread yet: wait for the send path to be ready before firing.
    if (transportsQuery.data?.status !== 200) return;

    // Latch before firing: a failed create/stream surfaces the standard error
    // state, and silently retrying it on the next visit is never what a hire
    // wants.
    firedRef.current = true;
    markKickedOff(expertId);
    onKickoffRef.current();
  }, [shouldKickoff, expertId, sessionsQuery.data, transportsQuery.data]);
}
