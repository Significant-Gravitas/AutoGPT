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

type Deferred = "sync" | "recover";
type CatchUp = Record<Deferred, () => Promise<void>>;

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
  const holdsGuard = useRef(false);
  const deferred = useRef<Deferred | null>(null);

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

  // A sync or recovery refused because a load holds `isSyncing` runs once the
  // load is done. A sync wins over a recovery: it reloads everything anyway.
  function deferUntilLoaded(action: Deferred) {
    if (!holdsGuard.current) return false;
    if (action === "sync" || deferred.current === null) {
      deferred.current = action;
    }
    return true;
  }

  function runDeferred(catchUp: CatchUp) {
    const action = deferred.current;
    deferred.current = null;
    if (!isMountedRef.current || action === null) return;
    void catchUp[action]().catch(() => undefined);
  }

  // Held under `isSyncing` like a sync, so a send or a resume can't land
  // between the fetch and the rebuild. Messages this screen streamed have SDK
  // or server ids, never the hydrated `${sessionID}-` prefix, so they stay.
  async function loadEarlier(catchUp: CatchUp) {
    const before = page.oldestSequence;
    if (!page.hasEarlier || before === null) return;
    if (isSyncingRef.current || isChatBusy()) return;
    isSyncingRef.current = true;
    holdsGuard.current = true;
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
      holdsGuard.current = false;
      if (isMountedRef.current) setIsLoadingEarlier(false);
      runDeferred(catchUp);
    }
  }

  return {
    hasEarlier: page.hasEarlier,
    isLoadingEarlier,
    loadEarlier,
    showServerMessages,
    deferUntilLoaded,
  };
}
