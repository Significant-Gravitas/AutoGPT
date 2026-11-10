import { Chat, useChat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useState } from "react";
import { AppState } from "react-native";

import {
  cancelSession,
  chatKeys,
  getSession,
  type SessionDetail,
} from "@/lib/api/chat";
import { queryClient } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import { toUIMessages } from "./convertMessages";
import { takePendingFirstMessage } from "./pendingFirstMessage";
import { createChatTransport } from "./transport";

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
      }),
  );
  const { messages, status, error, setMessages } = useChat({
    chat,
    throttle: 50,
  });
  const isBusy = status === "submitted" || status === "streaming";

  // The stream route replays a running turn from its first chunk, so the
  // local copy of that turn goes before resuming or its text shows twice.
  async function syncWithServer() {
    await chat.stop();
    const fresh = await queryClient.fetchQuery({
      queryKey: chatKeys.session(sessionID),
      queryFn: () => getSession(sessionID),
      staleTime: 0,
    });
    setMessages(
      toUIMessages(sessionID, fresh.messages, {
        activeTurnStartedAt: fresh.active_stream?.started_at,
      }),
    );
    if (fresh.active_stream) await chat.resumeStream();
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
    return () => subscription.remove();
  });

  function send(text: string) {
    if (isBusy) return;
    void chat.sendMessage({ text });
  }

  async function stop() {
    await chat.stop();
    await cancelSession(sessionID).catch(() => undefined);
  }

  return { messages, status, error, isBusy, send, stop };
}
