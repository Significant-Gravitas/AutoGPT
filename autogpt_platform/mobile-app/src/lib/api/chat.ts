import type { QueryClient } from "@tanstack/react-query";

import { apiRequest } from "./client";

export interface SessionSummary {
  id: string;
  created_at: string;
  updated_at: string;
  title: string | null;
  is_processing: boolean;
  is_pinned: boolean;
  expert_id: string | null;
}

export interface SessionDetail {
  id: string;
  messages: unknown[];
  active_stream: { turn_id: string; started_at?: string | null } | null;
  expert_id: string | null;
  has_more_messages: boolean;
  oldest_sequence: number | null;
}

interface SessionList {
  sessions: SessionSummary[];
  total: number;
}

interface SuggestedPrompts {
  themes: { name: string; prompts: string[] }[];
}

// Session ids are UUIDs (`@default(uuid())`, `uuid4`/`uuid5` in the backend).
// The route param comes from deep links too, so anything else is refused.
const SESSION_ID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isSessionID(value: unknown): value is string {
  return typeof value === "string" && SESSION_ID_RE.test(value);
}

export const chatKeys = {
  sessions: ["chat", "sessions"] as const,
  session: (sessionID: string) => ["chat", "session", sessionID] as const,
  suggestedPrompts: ["chat", "suggested-prompts"] as const,
};

/**
 * A chat opens from its cached copy at once; the thread then checks it with
 * the server itself (it reads the detail once, when it mounts), so the query
 * never refetches on mount. A copy younger than this is used as it is.
 */
export const sessionQueryOptions = {
  staleTime: 15_000,
  refetchOnMount: false,
} as const;

/** Starts loading a chat as a finger lands on its row. */
export function prefetchSession(queryClient: QueryClient, sessionID: string) {
  void queryClient.prefetchQuery({
    queryKey: chatKeys.session(sessionID),
    queryFn: () => getSession(sessionID),
    ...sessionQueryOptions,
  });
}

/** A chat made just now has no messages, so it can open without a fetch. */
export function emptySession(sessionID: string, expertID?: string): SessionDetail {
  return {
    id: sessionID,
    messages: [],
    active_stream: null,
    expert_id: expertID ?? null,
    has_more_messages: false,
    oldest_sequence: null,
  };
}

interface ListSessionsOptions {
  expertID?: string;
  limit?: number;
  pinnedFirst?: boolean;
}

export function listSessionsPage({
  expertID,
  limit = 50,
  pinnedFirst,
}: ListSessionsOptions = {}) {
  return apiRequest<SessionList>("/api/chat/sessions", {
    query: { limit, expert_id: expertID, pinned_first: pinnedFirst },
  });
}

export async function listSessions(options: ListSessionsOptions = {}) {
  const data = await listSessionsPage(options);
  return data.sessions;
}

interface GetSessionOptions {
  beforeSequence?: number;
}

export function getSession(
  sessionID: string,
  { beforeSequence }: GetSessionOptions = {},
) {
  return apiRequest<SessionDetail>(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}`,
    { query: { limit: 200, before_sequence: beforeSequence } },
  );
}

export function createSession(expertID?: string) {
  return apiRequest<{ id: string }>("/api/chat/sessions", {
    method: "POST",
    body: expertID ? { expert_id: expertID } : {},
  });
}

export function cancelSession(sessionID: string) {
  return apiRequest<{ cancelled: boolean }>(
    `/api/chat/sessions/${encodeURIComponent(sessionID)}/cancel`,
    { method: "POST" },
  );
}

export async function getSuggestedPrompts() {
  const data = await apiRequest<SuggestedPrompts>(
    "/api/chat/suggested-prompts",
  );
  return data.themes.flatMap((theme) => theme.prompts);
}
