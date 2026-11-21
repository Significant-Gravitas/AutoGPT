import { useMemo } from "react";
import {
  useGetV2GetSharedChat,
  useGetV2GetSharedChatMessages,
} from "@/app/api/__generated__/endpoints/chat/chat";
import { convertChatSessionMessagesToUiMessages } from "@/app/(platform)/copilot/helpers/convertChatSessionToUiMessages";
import { isDeadLinkError, retryUnlessDeadLink } from "@/lib/share/dead-link";
import { sharedChatFileUrl } from "@/lib/share/routes";

const PAGE_SIZE = 200;

export function useSharedChatPage(token: string) {
  const sessionQuery = useGetV2GetSharedChat(token, {
    query: {
      retry: retryUnlessDeadLink,
      select: (res) => (res.status === 200 ? res.data : undefined),
    },
  });

  const messagesQuery = useGetV2GetSharedChatMessages(
    token,
    { limit: PAGE_SIZE },
    {
      query: {
        enabled: !!sessionQuery.data,
        retry: retryUnlessDeadLink,
        select: (res) => (res.status === 200 ? res.data : undefined),
      },
    },
  );

  // isPending, since a read paused while the viewer is offline is pending but
  // not loading. The messages read waits for the session, so it only counts
  // once there is one.
  const isLoading =
    sessionQuery.isPending || (!!sessionQuery.data && messagesQuery.isPending);
  // Either read can find the link dead, since the owner can stop sharing
  // between them. Any other failure (a 5xx mid-deploy, a network blip) says
  // nothing about the link, so the page offers a retry instead.
  const isDeadLink =
    isDeadLinkError(sessionQuery.error) ||
    isDeadLinkError(messagesQuery.error) ||
    (sessionQuery.isSuccess && !sessionQuery.data);

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

  // Only the read that failed: the messages read follows the session on its
  // own once there is one.
  function retry() {
    if (sessionQuery.isError) sessionQuery.refetch();
    if (messagesQuery.isError) messagesQuery.refetch();
  }

  return {
    session: sessionQuery.data,
    rawMessages: messagesQuery.data?.messages ?? [],
    uiMessages: ui.uiMessages,
    turnStats: ui.turnStats,
    hasMore: messagesQuery.data?.has_more ?? false,
    isLoading,
    isDeadLink,
    messagesError: messagesQuery.isError,
    retry,
  };
}
