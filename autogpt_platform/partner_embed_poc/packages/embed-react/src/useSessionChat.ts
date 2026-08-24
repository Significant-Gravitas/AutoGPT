import { Chat, useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useRef, useState } from "react";

import type { AccessTokenProvider } from "./api";
import { createEmbeddedTransport } from "./transport";

interface UseSessionChatArgs {
  apiBaseURL: string;
  sessionID: string;
  initialMessages: UIMessage[];
  getAccessToken: AccessTokenProvider;
  onFinish: () => void;
}

export function useSessionChat({
  apiBaseURL,
  sessionID,
  initialMessages,
  getAccessToken,
  onFinish,
}: UseSessionChatArgs) {
  const tokenProviderRef = useRef(getAccessToken);
  const onFinishRef = useRef(onFinish);
  tokenProviderRef.current = getAccessToken;
  onFinishRef.current = onFinish;

  const [chat] = useState(
    () =>
      new Chat<UIMessage>({
        id: sessionID,
        messages: initialMessages,
        transport: createEmbeddedTransport({
          apiBaseURL,
          sessionID,
          getAccessToken: () => tokenProviderRef.current(),
        }),
        onFinish: () => onFinishRef.current(),
      }),
  );

  return useChat({ chat });
}
