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
import { SignedOutError } from "@/lib/auth/errors";
import { queryClient, signOut } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

import { toUIMessages } from "./convertMessages";
import { takePendingFirstMessage } from "./pendingFirstMessage";
import { chatErrorMessage } from "./chatErrors";
import {
  isRecoverableChatError,
  lastMessageIsUser,
  messagesInWindow,
  turnLanded,
} from "./recovery";
import { createChatTransport } from "./transport";
import { useEarlierMessages } from "./useEarlierMessages";

const RECOVERY_DELAY_MS = 1500;
const MAX_RECOVERIES = 3;

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
        onFinish: ({ isError }) => {
          if (!isError) recoveries.current = 0;
          void queryClient.invalidateQueries({ queryKey: chatKeys.sessions });
        },
        // A dead session goes back to Login instead of a stuck error line.
        onError: (chatError) => {
          if (chatError instanceof SignedOutError) void signOut();
          else if (isRecoverableChatError(chatError))
            void recover().catch(() => undefined);
        },
      }),
  );
  const { messages, status, error, setMessages } = useChat({
    chat,
    throttle: 50,
  });
  const isSyncing = useRef(false);
  const isMounted = useRef(true);
  const recoveries = useRef(0);
  const isBusy = status === "submitted" || status === "streaming";
  const earlier = useEarlierMessages({
    sessionID,
    detail,
    chat,
    setMessages,
    isSyncingRef: isSyncing,
    isMountedRef: isMounted,
  });

  function isChatBusy() {
    return chat.status === "submitted" || chat.status === "streaming";
  }

  // The stream route replays a running turn from its first chunk, so the
  // local copy of that turn goes before resuming or its text shows twice.
  // Read straight from the API: a cache write would outlive the screen.
  // Fetch first: if that fails, the live stream is left as it was.
  async function syncWithServer() {
    if (isSyncing.current) {
      earlier.deferUntilLoaded("sync");
      return;
    }
    isSyncing.current = true;
    let fresh: SessionDetail;
    try {
      fresh = await getSession(sessionID);
      if (!isMounted.current) return;
      await chat.stop();
      if (!isMounted.current) return;
      earlier.showServerMessages(fresh);
      chat.clearError();
    } finally {
      isSyncing.current = false;
    }
    if (fresh.active_stream && !isChatBusy()) await resumeTurn();
  }

  // resumeStream resolves once the replayed stream ends, or at once when the
  // turn already finished (the server answers 204). In that second case the
  // reply was left out of the hydrated messages, so load it. Callers release
  // `isSyncing` first; it is held here only for the fetch, so a send made
  // meanwhile is refused instead of wiped.
  async function resumeTurn() {
    if (!isMounted.current) return;
    await chat.resumeStream();
    if (!isMounted.current || isSyncing.current) return;
    if (isChatBusy() || !lastMessageIsUser(chat.messages)) return;
    isSyncing.current = true;
    try {
      const fresh = await getSession(sessionID);
      if (!isMounted.current || isChatBusy() || fresh.active_stream) return;
      earlier.showServerMessages(fresh);
    } finally {
      isSyncing.current = false;
    }
  }

  // A stream that drops in the foreground: if the turn reached the server,
  // show what it has and pick the stream up again.
  async function recover() {
    if (recoveries.current >= MAX_RECOVERIES) return;
    recoveries.current += 1;
    await new Promise((resolve) => setTimeout(resolve, RECOVERY_DELAY_MS));
    if (isSyncing.current && earlier.deferUntilLoaded("recover")) {
      recoveries.current -= 1;
      return;
    }
    if (!isMounted.current || isSyncing.current || isChatBusy()) return;
    isSyncing.current = true;
    let fresh: SessionDetail;
    try {
      fresh = await getSession(sessionID); // throws offline: the error line stays
      if (!isMounted.current || isChatBusy()) return;
      // Not taken, or taken but unanswered: keep the message and the error.
      const local = messagesInWindow(
        sessionID,
        chat.messages,
        fresh.oldest_sequence,
      );
      if (!turnLanded(fresh.messages, local, fresh.active_stream)) return;
      earlier.showServerMessages(fresh);
      chat.clearError();
    } finally {
      isSyncing.current = false;
    }
    if (fresh.active_stream) await resumeTurn();
  }

  useMountEffect(() => {
    isMounted.current = true;
    const first = takePendingFirstMessage(sessionID);
    if (first) void chat.sendMessage({ text: first });
    else if (detail.active_stream) void resumeTurn().catch(() => undefined);

    let wasBackgrounded = false;
    const subscription = AppState.addEventListener("change", (state) => {
      if (state === "background") wasBackgrounded = true;
      if (state !== "active" || !wasBackgrounded) return;
      wasBackgrounded = false;
      void syncWithServer().catch(() => undefined);
    });
    return () => {
      isMounted.current = false;
      subscription.remove();
      // Only the local stream: the turn keeps running on the server.
      void chat.stop();
    };
  });

  // True once the SDK has the message; it keeps it shown if the stream fails.
  function send(text: string) {
    if (isSyncing.current || isChatBusy()) return false;
    recoveries.current = 0;
    void chat.sendMessage({ text });
    return true;
  }

  function loadEarlier() {
    return earlier.loadEarlier({ sync: syncWithServer, recover });
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
    hasEarlier: earlier.hasEarlier,
    isLoadingEarlier: earlier.isLoadingEarlier,
    loadEarlier,
  };
}
