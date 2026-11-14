import type { SessionSummary } from "@/lib/api/chat";
import type { Expert } from "@/lib/api/experts";
import { palette } from "@/lib/expertPalette";
import { OTTO_AVATAR_URL, OTTO_NAME, OTTO_ROLE, OTTO_TINT } from "@/lib/otto";
import { avatarSource, avatarTint, roleLine } from "@/features/team/helpers";

// The list route's maximum page size.
export const DRAWER_SESSION_LIMIT = 100;

export interface ChatGroup {
  key: string;
  label: string;
  role: string | null;
  avatar: string | null;
  tint: string;
  isOtto: boolean;
  sessions: SessionSummary[];
  /** The group's most recently updated chat. */
  latest: SessionSummary;
  isBusy: boolean;
}

export interface ChatMatch {
  session: SessionSummary;
  group: ChatGroup;
}

// Same order as the web sidebar: Otto first, then each expert in the order
// their most recent chat appears in the list.
export function groupChatsByExpert(
  sessions: SessionSummary[],
  experts: Expert[],
): ChatGroup[] {
  const byExpert = new Map<string | null, SessionSummary[]>();
  for (const session of sessions) {
    const key = session.expert_id ?? null;
    const bucket = byExpert.get(key);
    if (bucket) bucket.push(session);
    else byExpert.set(key, [session]);
  }

  const expertsByID = new Map(experts.map((expert) => [expert.id, expert]));
  const groups = [...byExpert.entries()].map(([expertID, grouped]) =>
    toGroup(expertID, grouped, expertsByID),
  );
  return [
    ...groups.filter((group) => group.isOtto),
    ...groups.filter((group) => !group.isOtto),
  ];
}

// Past the first page the drawer can't know a group's real count.
export function chatCountLabel(count: number, isTruncated: boolean) {
  return isTruncated ? `${count}+` : String(count);
}

export function chatTitle(session: SessionSummary) {
  return session.title || "New chat";
}

/** Loaded chats whose title or expert name holds the query, newest first. */
export function searchChats(groups: ChatGroup[], query: string): ChatMatch[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [];
  return groups
    .flatMap((group) => group.sessions.map((session) => ({ session, group })))
    .filter(
      ({ session, group }) =>
        (session.title ?? "").toLowerCase().includes(needle) ||
        group.label.toLowerCase().includes(needle),
    )
    .sort((a, b) => updatedAt(b.session) - updatedAt(a.session));
}

function updatedAt(session: SessionSummary) {
  const time = Date.parse(session.updated_at);
  return Number.isNaN(time) ? 0 : time;
}

function latestOf(sessions: SessionSummary[]) {
  return sessions.reduce((latest, session) =>
    updatedAt(session) > updatedAt(latest) ? session : latest,
  );
}

function toGroup(
  expertID: string | null,
  sessions: SessionSummary[],
  expertsByID: Map<string, Expert>,
): ChatGroup {
  const shared = {
    sessions,
    latest: latestOf(sessions),
    isBusy: sessions.some((session) => session.is_processing),
  };
  if (!expertID) {
    return {
      key: "otto",
      label: OTTO_NAME,
      role: OTTO_ROLE,
      avatar: OTTO_AVATAR_URL,
      tint: OTTO_TINT,
      isOtto: true,
      ...shared,
    };
  }
  const expert = expertsByID.get(expertID);
  return {
    key: expertID,
    label: expert?.name || "Expert",
    role: expert ? roleLine(expert) : null,
    avatar: expert ? avatarSource(expert) : null,
    tint: expert ? avatarTint(expert) : palette.general.hex,
    isOtto: false,
    ...shared,
  };
}
