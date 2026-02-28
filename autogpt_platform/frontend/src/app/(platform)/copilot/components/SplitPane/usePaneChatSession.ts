/**
 * A version of useChatSession that takes sessionId/setSessionId as props
 * instead of syncing to URL query state. This allows multiple independent
 * chat sessions in a split-pane layout.
 */

import {
  getGetV2GetSessionQueryKey,
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { toast } from "@/components/molecules/Toast/use-toast";
import * as Sentry from "@sentry/nextjs";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef } from "react";
import { convertChatSessionMessagesToUiMessages } from "../../helpers/convertChatSessionToUiMessages";

interface UsePaneChatSessionArgs {
  sessionId: string | null;
  setSessionId: (id: string | null) => void;
}

export function usePaneChatSession({
  sessionId,
  setSessionId,
}: UsePaneChatSessionArgs) {
  const queryClient = useQueryClient();

  const sessionQuery = useGetV2GetSession(sessionId ?? "", {
    query: {
      enabled: !!sessionId,
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true,
    },
  });

  // Invalidate cache when navigating away from a session
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

  const hasActiveStream = useMemo(() => {
    if (sessionQuery.data?.status !== 200) return false;
    return !!sessionQuery.data.data.active_stream;
  }, [sessionQuery.data]);

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
        throw error;
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
