import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { format, formatDistanceToNow, isToday } from "date-fns";

export function filterVisibleSessions(
  sessions: SessionSummaryResponse[],
): SessionSummaryResponse[] {
  return sessions.filter(
    (session) => session.updated_at !== session.created_at,
  );
}

export function getSessionTitle(session: SessionSummaryResponse): string {
  if (session.title) return session.title;
  const isNewSession = session.updated_at === session.created_at;
  if (isNewSession) {
    const createdDate = new Date(session.created_at);
    if (isToday(createdDate)) {
      return "Today";
    }
    return format(createdDate, "MMM d, yyyy");
  }
  return "Untitled Chat";
}

export function getSessionUpdatedLabel(
  session: SessionSummaryResponse,
): string {
  if (!session.updated_at) return "";
  return formatDistanceToNow(new Date(session.updated_at), { addSuffix: true });
}
