import type { Chat } from "@ai-sdk/react";
import type { UIMessage } from "ai";
import { useRef, useState, type RefObject } from "react";

import { getSession, type SessionDetail } from "@/lib/api/chat";

import { toUIMessages } from "./convertMessages";

interface Args {
  sessionID: string;
  detail: SessionDetail;
  chat: Chat<UIMessage>;
  setMessages: (messages: UIMessage[]) => void;
  isSyncingRef: RefObject<boolean>;
  isMountedRef: RefObject<boolean>;
}

function pageOf(session: SessionDetail) {
  return {
    hasEarlier: session.has_more_messages && session.oldest_sequence !== null,
    oldestSequence: session.oldest_sequence,
  };
}

// Owns the stored rows behind the hydrated messages, oldest first, so a page
// of older rows is converted together with the newer ones: an assistant
// bubble or a tool result that straddles the page edge comes out whole.
export function useEarlierMessages({
  sessionID,
  detail,
  chat,
  setMessages,
  isSyncingRef,
  isMountedRef,
}: Args) {
  const rows = useRef(detail.messages);
  const activeTurnStartedAt = useRef(detail.active_stream?.started_at);
  const [page, setPage] = useState(() => pageOf(detail));
  const [isLoadingEarlier, setIsLoadingEarlier] = useState(false);

  function isChatBusy() {
    return chat.status === "submitted" || chat.status === "streaming";
  }

  function hydrate() {
    return toUIMessages(sessionID, rows.current, {
      activeTurnStartedAt: activeTurnStartedAt.current,
    });
  }

  function showServerMessages(fresh: SessionDetail) {
    rows.current = fresh.messages;
    activeTurnStartedAt.current = fresh.active_stream?.started_at;
    setPage(pageOf(fresh));
    setMessages(hydrate());
  }

  // Held under `isSyncing` like a sync, so a send or a resume can't land
  // between the fetch and the rebuild. Messages this screen streamed have SDK
  // or server ids, never the hydrated `${sessionID}-` prefix, so they stay.
  async function loadEarlier() {
    const before = page.oldestSequence;
    if (!page.hasEarlier || before === null) return;
    if (isSyncingRef.current || isChatBusy()) return;
    isSyncingRef.current = true;
    setIsLoadingEarlier(true);
    try {
      const older = await getSession(sessionID, { beforeSequence: before });
      if (!isMountedRef.current || isChatBusy()) return;
      rows.current = [...older.messages, ...rows.current];
      setPage({
        hasEarlier: older.has_more_messages && older.messages.length > 0,
        oldestSequence: older.oldest_sequence ?? before,
      });
      const live = chat.messages.filter(
        (message) => !message.id.startsWith(`${sessionID}-`),
      );
      setMessages([...hydrate(), ...live]);
    } finally {
      isSyncingRef.current = false;
      if (isMountedRef.current) setIsLoadingEarlier(false);
    }
  }

  return {
    hasEarlier: page.hasEarlier,
    isLoadingEarlier,
    loadEarlier,
    showServerMessages,
  };
}
