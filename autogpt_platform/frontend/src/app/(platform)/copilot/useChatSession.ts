import {
  getGetV2ListSessionsQueryKey,
  useGetV2GetSession,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { useQueryClient } from "@tanstack/react-query";
import { parseAsString, useQueryState } from "nuqs";
import { useMemo } from "react";
import { convertChatSessionMessagesToUiMessages } from "./helpers/convertChatSessionToUiMessages";

export function useChatSession() {
  const [sessionId, setSessionId] = useQueryState("sessionId", parseAsString);
  const queryClient = useQueryClient();

  const sessionQuery = useGetV2GetSession(sessionId ?? "", {
    query: {
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      refetchOnReconnect: false,
    },
  });

  // Memoize so the effect in useCopilotPage doesn't infinite-loop on a new
  // array reference every render. Re-derives only when query data changes.
  const hydratedMessages = useMemo(() => {
    if (sessionQuery.data?.status !== 200 || !sessionId) return undefined;
    return convertChatSessionMessagesToUiMessages(
      sessionId,
      sessionQuery.data.data.messages ?? [],
    );
  }, [sessionQuery.data, sessionId]);

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
    const response = await createSessionMutation();
    if (response.status !== 200 || !response.data?.id) {
      throw new Error("Failed to create session");
    }
    return response.data.id;
  }

  return {
    sessionId,
    setSessionId,
    hydratedMessages,
    isLoadingSession: sessionQuery.isLoading,
    createSession,
    isCreatingSession,
  };
}
