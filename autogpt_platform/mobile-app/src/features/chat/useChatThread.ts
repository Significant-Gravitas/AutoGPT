import { Chat, useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useRef, useState } from "react";
import { AppState } from "react-native";

import {
  cancelSession,
  chatKeys,
  getSession,
  type SessionDetail,
} from "@/lib/api/chat";
import { SignedOutError } from "@/lib/auth/token";
import { queryClient, signOut } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import { toUIMessages } from "./convertMessages";
import { takePendingFirstMessage } from "./pendingFirstMessage";
import { chatErrorMessage, createChatTransport } from "./transport";

interface Args {
  sessionID: string;
  detail: SessionDetail;
}

export function useChatThread({ sessionID, detail }: Args) {
  const [chat] = useState(
    () =>
      new Chat<UIMessage>({
        id: sessionID,
        messages: toUIMessages(sessionID, detail.messages, {
          activeTurnStartedAt: detail.active_stream?.started_at,
        }),
        transport: createChatTransport(sessionID),
        onFinish: () =>
          void queryClient.invalidateQueries({ queryKey: chatKeys.sessions }),
        // A dead session goes back to Login instead of a stuck error line.
        onError: (chatError) => {
          if (chatError instanceof SignedOutError) void signOut();
        },
      }),
  );
  const { messages, status, error, setMessages } = useChat({
    chat,
    throttle: 50,
  });
  const isSyncing = useRef(false);
  const isBusy = status === "submitted" || status === "streaming";

  function isChatBusy() {
    return chat.status === "submitted" || chat.status === "streaming";
  }

  // The stream route replays a running turn from its first chunk, so the
  // local copy of that turn goes before resuming or its text shows twice.
  // Read straight from the API: a cache write would outlive the screen.
  // Fetch first: if that fails, the live stream is left as it was.
  async function syncWithServer() {
    if (isSyncing.current) return;
    isSyncing.current = true;
    let fresh: SessionDetail;
    try {
      fresh = await getSession(sessionID);
      await chat.stop();
      setMessages(
        toUIMessages(sessionID, fresh.messages, {
          activeTurnStartedAt: fresh.active_stream?.started_at,
        }),
      );
      chat.clearError();
    } finally {
      isSyncing.current = false;
    }
    if (fresh.active_stream && !isChatBusy()) await chat.resumeStream();
  }

  useMountEffect(() => {
    const first = takePendingFirstMessage(sessionID);
    if (first) void chat.sendMessage({ text: first });
    else if (detail.active_stream) void chat.resumeStream();

    let wasBackgrounded = false;
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "background") wasBackgrounded = true;
      if (state !== "active" || !wasBackgrounded) return;
      wasBackgrounded = false;
      void syncWithServer().catch(() => undefined);
    });
    return () => {
      subscription.remove();
      // Only the local stream: the turn keeps running on the server.
      void chat.stop();
    };
  });

  // True once the SDK has the message; it keeps it shown if the stream fails.
  function send(text: string) {
    if (isSyncing.current || isChatBusy()) return false;
    void chat.sendMessage({ text });
    return true;
  }

  async function stop() {
    await chat.stop();
    await cancelSession(sessionID).catch(() => undefined);
  }

  return {
    messages,
    status,
    errorMessage: chatErrorMessage(error),
    isBusy,
    send,
    stop,
  };
}
