export interface SearchSession {
  id: string;
  title?: string | null;
  updated_at: string;
  is_processing?: boolean;
  chat_status?: string | null;
  source_platform?: string | null;
}

export interface HighlightPart {
  text: string;
  isMatch: boolean;
}

const EMPTY_QUERY_LIMIT = 10;
const ACTIVE_QUERY_LIMIT = 20;

function getSessionTimestamp(session: SearchSession) {
  const timestamp = new Date(session.updated_at).getTime();
  return Number.isNaN(timestamp) ? 0 : timestamp;
}

function getTitle(session: SearchSession) {
  return session.title || "Untitled chat";
}

export function filterSessions(
  sessions: SearchSession[],
  query: string,
): SearchSession[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  const sortedSessions = [...sessions].sort(
    (a, b) => getSessionTimestamp(b) - getSessionTimestamp(a),
  );

  if (!normalizedQuery) {
    return sortedSessions.slice(0, EMPTY_QUERY_LIMIT);
  }

  return sortedSessions
    .filter((session) =>
      getTitle(session).toLocaleLowerCase().includes(normalizedQuery),
    )
    .slice(0, ACTIVE_QUERY_LIMIT);
}

export function highlightMatch(title: string, query: string): HighlightPart[] {
  const normalizedQuery = query.trim().toLocaleLowerCase();
  if (!normalizedQuery) return [{ text: title, isMatch: false }];

  const matchIndex = title.toLocaleLowerCase().indexOf(normalizedQuery);
  if (matchIndex === -1) return [{ text: title, isMatch: false }];

  const matchEnd = matchIndex + normalizedQuery.length;
  return [
    { text: title.slice(0, matchIndex), isMatch: false },
    { text: title.slice(matchIndex, matchEnd), isMatch: true },
    { text: title.slice(matchEnd), isMatch: false },
  ].filter((part) => part.text.length > 0);
}

export function formatRelativeDate(dateString: string, baseDate = new Date()) {
  const date = new Date(dateString);
  if (Number.isNaN(date.getTime())) return "";

  const diffMs = baseDate.getTime() - date.getTime();
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffDays <= 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return `${diffDays} days ago`;

  const day = date.getDate();
  const ordinal =
    day % 10 === 1 && day !== 11
      ? "st"
      : day % 10 === 2 && day !== 12
        ? "nd"
        : day % 10 === 3 && day !== 13
          ? "rd"
          : "th";
  const month = date.toLocaleDateString("en-US", { month: "short" });
  const year = date.getFullYear();

  return `${day}${ordinal} ${month} ${year}`;
}

export function getSessionTitle(session: SearchSession) {
  return getTitle(session);
}
