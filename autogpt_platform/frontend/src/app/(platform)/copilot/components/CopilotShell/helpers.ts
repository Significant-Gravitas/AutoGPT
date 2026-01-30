import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { format, formatDistanceToNow, isToday } from "date-fns";

export function convertSessionDetailToSummary(session: SessionDetailResponse) {
  return {
    id: session.id,
    created_at: session.created_at,
    updated_at: session.updated_at,
    title: undefined,
  };
}

export function filterVisibleSessions(sessions: SessionSummaryResponse[]) {
  const fiveMinutesAgo = Date.now() - 5 * 60 * 1000;
  return sessions.filter((session) => {
    const hasBeenUpdated = session.updated_at !== session.created_at;

    if (hasBeenUpdated) return true;

    const isRecentlyCreated =
      new Date(session.created_at).getTime() > fiveMinutesAgo;

    return isRecentlyCreated;
  });
}

export function getSessionTitle(session: SessionSummaryResponse) {
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

export function getSessionUpdatedLabel(session: SessionSummaryResponse) {
  if (!session.updated_at) return "";
  return formatDistanceToNow(new Date(session.updated_at), { addSuffix: true });
}

export function mergeCurrentSessionIntoList(
  accumulatedSessions: SessionSummaryResponse[],
  currentSessionId: string | null,
  currentSessionData: SessionDetailResponse | null | undefined,
  recentlyCreatedSessions?: Map<string, SessionSummaryResponse>,
) {
  const filteredSessions: SessionSummaryResponse[] = [];
  const addedIds = new Set<string>();

  if (accumulatedSessions.length > 0) {
    const visibleSessions = filterVisibleSessions(accumulatedSessions);

    if (currentSessionId) {
      const currentInAll = accumulatedSessions.find(
        (s) => s.id === currentSessionId,
      );
      if (currentInAll) {
        const isInVisible = visibleSessions.some(
          (s) => s.id === currentSessionId,
        );
        if (!isInVisible) {
          filteredSessions.push(currentInAll);
          addedIds.add(currentInAll.id);
        }
      }
    }

    for (const session of visibleSessions) {
      if (!addedIds.has(session.id)) {
        filteredSessions.push(session);
        addedIds.add(session.id);
      }
    }
  }

  if (currentSessionId && currentSessionData) {
    if (!addedIds.has(currentSessionId)) {
      const summarySession = convertSessionDetailToSummary(currentSessionData);
      filteredSessions.unshift(summarySession);
      addedIds.add(currentSessionId);
    }
  }

  if (recentlyCreatedSessions) {
    for (const [sessionId, sessionData] of recentlyCreatedSessions) {
      if (!addedIds.has(sessionId)) {
        filteredSessions.unshift(sessionData);
        addedIds.add(sessionId);
      }
    }
  }

  return filteredSessions;
}

export function getCurrentSessionId(searchParams: URLSearchParams) {
  return searchParams.get("sessionId");
}
