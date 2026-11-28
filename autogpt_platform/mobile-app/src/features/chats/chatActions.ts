import type { ActionOption } from "../../lib/actionMenu";
import type { SessionList, SessionSummary } from "../../lib/api/chat";

export type ChatAction = "rename" | "pin" | "delete";

// The title route's own limit.
const MAX_TITLE_LENGTH = 200;

export function chatActionOptions(session: SessionSummary): ActionOption<ChatAction>[] {
  return [
    { id: "rename", label: "Rename" },
    { id: "pin", label: session.is_pinned ? "Unpin" : "Pin" },
    { id: "delete", label: "Delete", isDestructive: true },
  ];
}

export function validateTitle(text: string): { title: string } | { error: string } {
  const title = text.trim();
  if (!title) return { error: "Give the chat a name." };
  if (title.length > MAX_TITLE_LENGTH) {
    return { error: `Keep the name to ${MAX_TITLE_LENGTH} characters or fewer.` };
  }
  return { title };
}

export function withRenamed(
  list: SessionList,
  sessionID: string,
  title: string | null,
): SessionList {
  return {
    ...list,
    sessions: list.sessions.map((session) =>
      session.id === sessionID ? { ...session, title } : session,
    ),
  };
}

/** Pinned chats first, then newest first, as the server orders the list. */
export function withPinned(
  list: SessionList,
  sessionID: string,
  isPinned: boolean,
): SessionList {
  const sessions = list.sessions.map((session) =>
    session.id === sessionID ? { ...session, is_pinned: isPinned } : session,
  );
  return { ...list, sessions: sessions.sort(compareSessions) };
}

export function withoutSession(list: SessionList, sessionID: string): SessionList {
  const sessions = list.sessions.filter((session) => session.id !== sessionID);
  const removed = list.sessions.length - sessions.length;
  return { sessions, total: Math.max(0, list.total - removed) };
}

export function restoreSession(
  list: SessionList,
  previous: SessionSummary,
  optimistic: SessionSummary | undefined,
): SessionList {
  const current = list.sessions.find((session) => session.id === previous.id);
  if (!optimistic) {
    if (current) return list;
    const sessions = [...list.sessions, previous].sort(compareSessions);
    return { ...list, sessions, total: list.total + 1 };
  }
  if (!current) return list;
  if (previous.title !== optimistic.title && current.title === optimistic.title) {
    return withRenamed(list, previous.id, previous.title);
  }
  if (
    previous.is_pinned !== optimistic.is_pinned &&
    current.is_pinned === optimistic.is_pinned
  ) {
    return withPinned(list, previous.id, previous.is_pinned);
  }
  return list;
}

function compareSessions(a: SessionSummary, b: SessionSummary) {
  if (a.is_pinned !== b.is_pinned) return a.is_pinned ? -1 : 1;
  return updatedAt(b) - updatedAt(a);
}

function updatedAt(session: SessionSummary) {
  const time = Date.parse(session.updated_at);
  return Number.isNaN(time) ? 0 : time;
}
