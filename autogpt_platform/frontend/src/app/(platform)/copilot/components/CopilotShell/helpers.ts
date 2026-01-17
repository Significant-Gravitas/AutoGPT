import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { formatDistanceToNow } from "date-fns";

export function filterVisibleSessions(
  sessions: SessionSummaryResponse[],
): SessionSummaryResponse[] {
  return sessions.filter(
    (session) => session.updated_at !== session.created_at,
  );
}

export function getSessionTitle(session: SessionSummaryResponse): string {
  return session.title || "Untitled Chat";
}

export function getSessionUpdatedLabel(
  session: SessionSummaryResponse,
): string {
  if (!session.updated_at) return "";
  return formatDistanceToNow(new Date(session.updated_at), { addSuffix: true });
}
