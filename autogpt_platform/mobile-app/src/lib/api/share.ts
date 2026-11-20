import { apiRequest } from "./client";

/** `ChatShareStateResponse` in backend/api/features/chat/share.py. */
export interface ChatShareState {
  is_shared: boolean;
  share_token: string | null;
  auto_share_executions: boolean;
  message_count: number;
  linked_run_count: number;
  file_count: number;
}

export interface ChatShare {
  share_url: string;
  share_token: string;
}

export const shareKeys = {
  state: (sessionID: string) => ["chat", sessionID, "share"] as const,
};

function sharePath(sessionID: string) {
  return `/api/chat/sessions/${encodeURIComponent(sessionID)}/share`;
}

export function getShareState(sessionID: string) {
  return apiRequest<ChatShareState>(`${sharePath(sessionID)}/state`);
}

export function enableSharing(sessionID: string, autoShareExecutions: boolean) {
  return apiRequest<ChatShare>(sharePath(sessionID), {
    method: "POST",
    body: { auto_share_executions: autoShareExecutions },
  });
}

export function disableSharing(sessionID: string) {
  return apiRequest<void>(sharePath(sessionID), { method: "DELETE" });
}
