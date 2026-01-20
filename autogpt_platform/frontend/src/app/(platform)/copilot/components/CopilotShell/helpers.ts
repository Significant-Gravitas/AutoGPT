import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { format, formatDistanceToNow, isToday } from "date-fns";

export function convertSessionDetailToSummary(
  session: SessionDetailResponse,
): SessionSummaryResponse {
  return {
    id: session.id,
    created_at: session.created_at,
    updated_at: session.updated_at,
    title: undefined,
  };
}

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

export function mergeCurrentSessionIntoList(
  accumulatedSessions: SessionSummaryResponse[],
  currentSessionId: string | null,
  currentSessionData: SessionDetailResponse | null | undefined,
): SessionSummaryResponse[] {
  const filteredSessions: SessionSummaryResponse[] = [];

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
        }
      }
    }

    filteredSessions.push(...visibleSessions);
  }

  if (currentSessionId && currentSessionData) {
    const isCurrentInList = filteredSessions.some(
      (s) => s.id === currentSessionId,
    );
    if (!isCurrentInList) {
      const summarySession = convertSessionDetailToSummary(currentSessionData);
      filteredSessions.unshift(summarySession);
    }
  }

  return filteredSessions;
}

export function getCurrentSessionId(
  searchParams: URLSearchParams,
): string | null {
  return searchParams.get("sessionId");
}

export function shouldAutoSelectSession(
  areAllSessionsLoaded: boolean,
  hasAutoSelectedSession: boolean,
  paramSessionId: string | null,
  visibleSessions: SessionSummaryResponse[],
  accumulatedSessions: SessionSummaryResponse[],
  isLoading: boolean,
  totalCount: number | null,
): {
  shouldSelect: boolean;
  sessionIdToSelect: string | null;
  shouldCreate: boolean;
} {
  if (!areAllSessionsLoaded || hasAutoSelectedSession) {
    return {
      shouldSelect: false,
      sessionIdToSelect: null,
      shouldCreate: false,
    };
  }

  if (paramSessionId) {
    return {
      shouldSelect: false,
      sessionIdToSelect: null,
      shouldCreate: false,
    };
  }

  if (visibleSessions.length > 0) {
    return {
      shouldSelect: true,
      sessionIdToSelect: visibleSessions[0].id,
      shouldCreate: false,
    };
  }

  if (accumulatedSessions.length === 0 && !isLoading && totalCount === 0) {
    return { shouldSelect: false, sessionIdToSelect: null, shouldCreate: true };
  }

  if (totalCount === 0) {
    return {
      shouldSelect: false,
      sessionIdToSelect: null,
      shouldCreate: false,
    };
  }

  return { shouldSelect: false, sessionIdToSelect: null, shouldCreate: false };
}

export function checkReadyToShowContent(
  areAllSessionsLoaded: boolean,
  paramSessionId: string | null,
  accumulatedSessions: SessionSummaryResponse[],
  isCurrentSessionLoading: boolean,
  currentSessionData: SessionDetailResponse | null | undefined,
  hasAutoSelectedSession: boolean,
): boolean {
  if (!areAllSessionsLoaded) return false;

  if (paramSessionId) {
    const sessionFound = accumulatedSessions.some(
      (s) => s.id === paramSessionId,
    );
    return (
      sessionFound ||
      (!isCurrentSessionLoading &&
        currentSessionData !== undefined &&
        currentSessionData !== null)
    );
  }

  return hasAutoSelectedSession;
}
