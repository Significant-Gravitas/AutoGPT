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
import { convertChatSessionMessagesToUiMessages } from "./helpers/convertChatSessionToUiMessages";

export function useChatSession() {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const queryClient = useQueryClient();

  const sessionQuery = useGetV2GetSession(sessionId ?? "", {
    query: {
      enabled: !!sessionId,
      staleTime: Infinity, // Manual invalidation on session switch
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
    },
  });

  // When the user navigates away from a session, invalidate its query cache.
  // useChat destroys its Chat instance on id change, so messages are lost.
  // Invalidating ensures the next visit fetches fresh data from the API
  // instead of hydrating from stale cache that's missing recent messages.
  const prevSessionIdRef = useRef(sessionId);

  useEffect(() => {
    const prev = prevSessionIdRef.current;
    prevSessionIdRef.current = sessionId;
    if (prev && prev !== sessionId) {
      queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(prev),
      });
    }
  }, [sessionId, queryClient]);

  // Expose active_stream info so the caller can trigger manual resume
  // after hydration completes (rather than relying on AI SDK's built-in
  // resume which fires before hydration).
  const hasActiveStream = useMemo(() => {
    if (sessionQuery.data?.status !== 200) return false;
    return !!sessionQuery.data.data.active_stream;
  }, [sessionQuery.data, sessionId]);

  // Memoize so the effect in useCopilotPage doesn't infinite-loop on a new
  // array reference every render. Re-derives only when query data changes.
  // When the session is complete (no active stream), mark dangling tool
  // calls as completed so stale spinners don't persist after refresh.
  const hydratedMessages = useMemo(() => {
    if (sessionQuery.data?.status !== 200 || !sessionId) return undefined;
    return convertChatSessionMessagesToUiMessages(
      sessionId,
      sessionQuery.data.data.messages ?? [],
      { isComplete: !hasActiveStream },
    );
  }, [sessionQuery.data, sessionId, hasActiveStream]);

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
      const response = await createSessionMutation();
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

  return {
    sessionId,
    setSessionId,
    hydratedMessages,
    hasActiveStream,
    isLoadingSession: sessionQuery.isLoading,
    isSessionError: sessionQuery.isError,
    createSession,
    isCreatingSession,
    refetchSession: sessionQuery.refetch,
  };
}
