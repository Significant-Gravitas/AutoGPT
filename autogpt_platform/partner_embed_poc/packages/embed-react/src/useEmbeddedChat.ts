import { Chat, useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useEffect, useRef, useState } from "react";

import { createEmbedSession, type AccessTokenProvider } from "./api";
import type { EmbeddedChatState } from "./chat-state";
import { createEmbeddedTransport } from "./transport";

interface UseEmbeddedChatArgs {
  apiBaseURL: string;
  getAccessToken: AccessTokenProvider;
}

function useEmbeddedChat({
  apiBaseURL,
  getAccessToken,
}: UseEmbeddedChatArgs): EmbeddedChatState {
  const tokenProviderRef = useRef(getAccessToken);
  tokenProviderRef.current = getAccessToken;
  const [chat, setChat] = useState<Chat<UIMessage> | null>(null);
  const [initializationError, setInitializationError] = useState<string | null>(
    null,
  );

  useEffect(() => {
    let cancelled = false;
    setChat(null);
    setInitializationError(null);

    async function initialize() {
      try {
        const session = await createEmbedSession(apiBaseURL, () =>
          tokenProviderRef.current(),
        );
        if (cancelled) return;
        setChat(
          new Chat<UIMessage>({
            id: session.id,
            transport: createEmbeddedTransport({
              apiBaseURL,
              sessionID: session.id,
              getAccessToken: () => tokenProviderRef.current(),
            }),
          }),
        );
      } catch (error) {
        if (cancelled) return;
        setInitializationError(
          error instanceof Error ? error.message : "Unable to initialize chat",
        );
      }
    }

    void initialize();
    return () => {
      cancelled = true;
    };
  }, [apiBaseURL]);

  const chatState = useChat(chat ? { chat } : { id: "partner-embed-pending" });
  return { ...chatState, initializationError, isInitialized: chat !== null };
}

export { useEmbeddedChat };
