import { useMemo } from "react";
import {
  useGetV2GetSharedChat,
  useGetV2GetSharedChatMessages,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { sharedChatFileUrl } from "@/lib/share/routes";

const PAGE_SIZE = 200;

// Retry transient failures, but not real 404s — a missing share is a
// permanent state and looping retries on it would just delay the error
// UI without changing the outcome.
function retryUnlessNotFound(failureCount: number, error: unknown): boolean {
  const status = (error as { status?: number } | null)?.status;
  if (status === 404) return false;
  return failureCount < 3;
}

export function useSharedChatPage(token: string) {
  const sessionQuery = useGetV2GetSharedChat(token, {
    query: {
      retry: retryUnlessNotFound,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });

  const messagesQuery = useGetV2GetSharedChatMessages(
    token,
    { limit: PAGE_SIZE },
    {
      query: {
        enabled: !!sessionQuery.data,
        retry: retryUnlessNotFound,
        select: (res) => (res.status === 200 ? res.data : undefined),
      },
    },
  );

  const isLoading = sessionQuery.isLoading || messagesQuery.isLoading;
  const isError = sessionQuery.isError || messagesQuery.isError;
  const rawError = sessionQuery.error || messagesQuery.error;
  const error = rawError instanceof Error ? rawError.message : undefined;

  // Convert the sanitized SharedChatMessage[] into the UIMessage[] shape
  // that the owner-side ChatMessagesContainer expects.  ``isComplete``
  // is hard-coded true because public shares are by definition a
  // finished snapshot — there's no live stream, so any tool-call without
  // a paired tool-row should render as completed (no spinner).
  const ui = useMemo(() => {
    if (!sessionQuery.data || !messagesQuery.data) {
      return {
        uiMessages: [],
        turnStats: undefined as
          | ReturnType<typeof convertChatSessionMessagesToUiMessages>["stats"]
          | undefined,
      };
    }
    const converted = convertChatSessionMessagesToUiMessages(
      sessionQuery.data.id,
      messagesQuery.data.messages,
      {
        isComplete: true,
        // Route file URLs through the public allowlist-gated download
        // endpoint so anonymous viewers can render attachments without
        // hitting the auth-protected workspace download.  The URL
        // shape and the renderer's matching regex
        // (``sharedChatFilePattern``) live together in
        // ``lib/share/routes.ts`` so they evolve together.
        fileUrlBuilder: (fileId) => sharedChatFileUrl(token, fileId),
      },
    );
    return { uiMessages: converted.messages, turnStats: converted.stats };
  }, [sessionQuery.data, messagesQuery.data, token]);

  return {
    session: sessionQuery.data,
    rawMessages: messagesQuery.data?.messages ?? [],
    uiMessages: ui.uiMessages,
    turnStats: ui.turnStats,
    hasMore: messagesQuery.data?.has_more ?? false,
    isLoading,
    isError,
    error,
    retry: () => {
      sessionQuery.refetch();
      messagesQuery.refetch();
    },
  };
}
