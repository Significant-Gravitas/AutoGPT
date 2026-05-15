import {
  getGetV2GetSessionQueryKey,
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useEffect, useMemo, useRef } from "react";
import {
  convertChatSessionMessagesToUiMessages,
  type TurnStatsMap,
} from "./helpers/convertChatSessionToUiMessages";
import { resolveSessionDryRun } from "./helpers";

interface UseChatSessionOptions {
  dryRun?: boolean;
}

export function useChatSession({ dryRun = false }: UseChatSessionOptions = {}) {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const queryClient = useQueryClient();

  const sessionQuery = useGetV2GetSession(sessionId ?? "", undefined, {
    query: {
      enabled: !!sessionId,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
    },
  });

  // When dry-run mode is toggled, discard the current session so the next
  // send creates a fresh one with the correct dry_run flag.  Sessions are
  // immutable once created: dry_run cannot be changed after the fact.
  const prevDryRunRef = useRef(dryRun);
  useEffect(() => {
    if (prevDryRunRef.current !== dryRun) {
      prevDryRunRef.current = dryRun;
      if (sessionId) {
        setSessionId(null);
      }
    }
  }, [dryRun, sessionId, setSessionId]);

  // Invalidate query cache on session switch.
  // useChat destroys its Chat instance on id change, so messages are lost.
  // We invalidate BOTH the old session (stale after leaving) and the new
  // session (may have been cached before the user sent their last message,
  // so active_stream and messages could be outdated). This guarantees a
  // fresh fetch that gives the resume effect accurate hasActiveStream state.
  const prevSessionIdRef = useRef(sessionId);

  useEffect(() => {
    const prev = prevSessionIdRef.current;
    prevSessionIdRef.current = sessionId;
    if (prev && prev !== sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(prev),
      });
    }
    if (sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(sessionId),
      });
    }
  }, [sessionId, queryClient]);

  const freshSessionData =
    !!sessionId && sessionQuery.data?.status === 200 && !sessionQuery.isFetching
      ? sessionQuery.data.data
      : null;

  // Expose active_stream info so the caller can trigger manual resume
  // after hydration completes (rather than relying on AI SDK's built-in
  // resume which fires before hydration).
  const hasActiveStream = useMemo(() => {
    return !!freshSessionData?.active_stream;
  }, [freshSessionData]);

  // Backend-reported start time of the active turn. Used to seed the
  // elapsed-time counter on mount so restored sessions show honest
  // "time since the backend started the turn" rather than "time since
  // this mount subscribed to the SSE".
  const activeStreamStartedAt = useMemo(() => {
    return freshSessionData?.active_stream?.started_at ?? null;
  }, [freshSessionData]);

  // Pagination metadata from the initial page load
  const hasMoreMessages = useMemo(() => {
    return !!freshSessionData?.has_more_messages;
  }, [freshSessionData]);

  const oldestSequence = useMemo(() => {
    return freshSessionData?.oldest_sequence ?? null;
  }, [freshSessionData]);

  // Memoize so the effect in useCopilotPage doesn't infinite-loop on a new
  // array reference every render. Re-derives only when query data changes.
  // When the session is complete (no active stream), mark dangling tool
  // calls as completed so stale spinners don't persist after refresh.
  const { hydratedMessages, historicalTurnStats } = useMemo(() => {
    if (!freshSessionData || !sessionId)
      return {
        hydratedMessages: undefined,
        historicalTurnStats: new Map() as TurnStatsMap,
      };
    const result = convertChatSessionMessagesToUiMessages(
      sessionId,
      freshSessionData.messages ?? [],
      { isComplete: !hasActiveStream },
    );
    return {
      hydratedMessages: result.messages,
      historicalTurnStats: result.stats,
    };
  }, [freshSessionData, sessionId, hasActiveStream]);

  const { mutateAsync: createSessionMutation, isPending: isCreatingSession } =
    usePostV2CreateSession({
      mutation: {
        onSuccess: (response) => {
          if (response.status === 200 && response.data?.id) {
            setSessionId(response.data.id);
            queryClient.invalidateQueries({
              queryKey: getGetV2ListSessionsQueryKey(),
            });
          }
        },
      },
    });

  async function createSession() {
    if (sessionId) return sessionId;
    try {
      const body = dryRun ? { data: { dry_run: true } } : { data: null };
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const response = await (createSessionMutation as any)(body);
      if (response.status !== 200 || !response.data?.id) {
        const error = new Error("Failed to create session");
        Sentry.captureException(error, {
          extra: { status: response.status },
        });
        toast({
          variant: "destructive",
          title: "Could not start a new chat session",
          description: "Please try again.",
        });
        throw error;
      }
      return response.data.id;
    } catch (error) {
      if (
        error instanceof Error &&
        error.message === "Failed to create session"
      ) {
        throw error; // already handled above
      }
      Sentry.captureException(error);
      toast({
        variant: "destructive",
        title: "Could not start a new chat session",
        description: "Please try again.",
      });
      throw error;
    }
  }

  // Raw messages from the initial page — exposed for cross-page
  // tool output matching by useLoadMoreMessages.
  const rawSessionMessages =
    freshSessionData?.messages != null
      ? ((freshSessionData.messages ?? []) as unknown[])
      : [];

  // The actual dry_run value stored in the session's metadata, read directly
  // from the API response. This reflects what the session was ACTUALLY created
  // with — not the user's current UI preference (isDryRun store).
  //
  // Design intent: the global isDryRun store is only used when creating NEW
  // sessions. Once a session exists, its dry_run flag is immutable and should
  // be read from here rather than from the store, which may have changed.
  const sessionDryRun = useMemo(
    () => (freshSessionData ? resolveSessionDryRun(sessionQuery.data) : false),
    [sessionQuery.data, freshSessionData],
  );

  const sessionChatStatus = (
    freshSessionData as { chat_status?: string } | undefined
  )?.chat_status;

  return {
    sessionId,
    setSessionId,
    hydratedMessages,
    rawSessionMessages,
    historicalTurnStats,
    hasActiveStream,
    activeStreamStartedAt,
    hasMoreMessages,
    oldestSequence,
    // Only treat the session as loading during the INITIAL fetch (no cached
    // data yet). Background refetches keep the input enabled — otherwise a
    // fill+Enter race can trigger handleSend while ``disabled`` briefly
    // flips back to ``true`` mid-refetch, silently dropping the message.
    isLoadingSession: sessionQuery.isLoading,
    isSessionError: sessionQuery.isError,
    createSession,
    isCreatingSession,
    refetchSession: sessionQuery.refetch,
    sessionDryRun,
    sessionChatStatus,
  };
}
