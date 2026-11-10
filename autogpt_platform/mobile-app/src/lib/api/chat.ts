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
}

interface SessionList {
  sessions: SessionSummary[];
  total: number;
}

interface SuggestedPrompts {
  themes: { name: string; prompts: string[] }[];
}

export const chatKeys = {
  sessions: ["chat", "sessions"] as const,
  session: (sessionID: string) => ["chat", "session", sessionID] as const,
  suggestedPrompts: ["chat", "suggested-prompts"] as const,
};

interface ListSessionsOptions {
  expertID?: string;
  limit?: number;
  pinnedFirst?: boolean;
}

export async function listSessions({
  expertID,
  limit = 50,
  pinnedFirst,
}: ListSessionsOptions = {}) {
  const data = await apiRequest<SessionList>("/api/chat/sessions", {
    query: { limit, expert_id: expertID, pinned_first: pinnedFirst },
  });
  return data.sessions;
}

export function getSession(sessionID: string) {
  return apiRequest<SessionDetail>(`/api/chat/sessions/${sessionID}`, {
    query: { limit: 200 },
  });
}

export function createSession(expertID?: string) {
  return apiRequest<{ id: string }>("/api/chat/sessions", {
    method: "POST",
    body: expertID ? { expert_id: expertID } : {},
  });
}

export function cancelSession(sessionID: string) {
  return apiRequest<{ cancelled: boolean }>(
    `/api/chat/sessions/${sessionID}/cancel`,
    { method: "POST" },
  );
}

export async function getSuggestedPrompts() {
  const data = await apiRequest<SuggestedPrompts>(
    "/api/chat/suggested-prompts",
  );
  return data.themes.flatMap((theme) => theme.prompts);
}
