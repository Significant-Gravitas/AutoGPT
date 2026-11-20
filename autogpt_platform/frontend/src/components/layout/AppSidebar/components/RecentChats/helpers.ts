import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import {
  flattenSessions,
  type SessionListInfiniteData,
} from "@/app/(platform)/copilot/useSessionList";
import type { SessionGroupResponse } from "@/app/api/__generated__/models/sessionGroupResponse";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Key, storage } from "@/services/storage/local-storage";

/** The newest chats every group shows, however old they are. */
export const GROUP_MIN_CHATS = 3;
/** Chats the sidebar shows before any "Show more", about what a tall
 *  portrait screen fits: past each group's minimum, the newest fill it, so
 *  the experts in use show the most. */
export const SIDEBAR_CHAT_LIMIT = 30;
/** The most chats a group opens on in the phone menu, where the whole budget
 *  would push every other group off the screen. A desktop sidebar has room
 *  for it. */
export const PHONE_GROUP_CHATS = 4;
/** Chats each press of a group's "Show more" adds. */
export const GROUP_SHOW_MORE_STEP = 10;
/** Chats per request when a group pages in its older chats; above the
 *  sidebar limit, so the first "Show more" needs one request. */
export const GROUP_PAGE_LIMIT = 40;

// The key Otto's own group is remembered under; experts use their id.
const OTTO_GROUP_KEY = "autopilot";

// Fired experts can't be addressed, so their group gets no new-chat entry:
// the deep link would only fall back to Otto. `new=1` keeps the page on the
// fresh new-task state: without it `useChatSession` adopts the expert's
// latest thread, which is exactly the chat the user is trying to leave.
export function getNewChatHref(
  expertId: string | null,
  activeExpertIds: ReadonlySet<string>,
) {
  if (!expertId) return "/home";
  if (!activeExpertIds.has(expertId)) return null;
  return `/home?expertId=${encodeURIComponent(expertId)}&new=1`;
}

// How long a new expert keeps their own sidebar entry while they have no chat.
export const NEW_EXPERT_WINDOW_MS = 7 * 24 * 60 * 60 * 1000;

/** Experts who joined in the last week and have no chat yet, newest first.
 *  Each gets a sidebar group leading to their first chat; once a chat exists
 *  their normal group takes over. */
export function getNewExpertsWithoutChats(
  activeExperts: ExpertIdentity[],
  expertIdsWithChats: (string | null | undefined)[],
  now: number,
) {
  const withChats = new Set(expertIdsWithChats);
  return activeExperts
    .filter((expert) => {
      const joined = expert.createdAt?.getTime();
      return (
        joined !== undefined &&
        now - joined < NEW_EXPERT_WINDOW_MS &&
        !withChats.has(expert.id)
      );
    })
    .sort(
      (a, b) => (b.createdAt?.getTime() ?? 0) - (a.createdAt?.getTime() ?? 0),
    );
}

/** The groups without the chats being deleted, each total lowered to match;
 *  a group left with no chats is dropped. Untouched groups keep their
 *  object. */
export function withoutDeletingSessions(
  groups: SessionGroupResponse[],
  deletingIds: string[],
) {
  if (!deletingIds.length) return groups;
  return groups.flatMap((group) => {
    const sessions = group.sessions.filter(
      (session) => !deletingIds.includes(session.id),
    );
    const removed = group.sessions.length - sessions.length;
    if (!removed) return [group];
    const total = group.total - removed;
    return total > 0 ? [{ ...group, total, sessions }] : [];
  });
}

/** How many chats a group opens on: every one the grouped read lists for it,
 *  or on a phone at most `PHONE_GROUP_CHATS`. */
export function openingChatCount(listed: number, isPhone: boolean) {
  return isPhone ? Math.min(listed, PHONE_GROUP_CHATS) : listed;
}

/** A group's chats, newest first: its preview, which the sidebar's poll keeps
 *  fresh, over the older pages read when "Show more" was pressed, each chat
 *  once even when two pages overlap. */
export function mergeGroupSessions(
  preview: SessionSummaryResponse[],
  older: SessionSummaryResponse[],
) {
  if (!older.length) return preview;
  const seen = new Set(preview.map((session) => session.id));
  const rest = older.filter((session) => {
    if (seen.has(session.id)) return false;
    seen.add(session.id);
    return true;
  });
  return [...preview, ...rest].sort(newestFirst);
}

/** Whether a group's older pages still line up with what the poll says the
 *  group holds: every page read at the same total, and the same chats at the
 *  top. A chat added or deleted since a page was read, even between two of
 *  the reads, shifts every offset after it, so a page repeats or skips one. */
export function olderPagesMatchGroup(
  pages: SessionListInfiniteData,
  preview: SessionSummaryResponse[],
  total: number,
) {
  const totals = pages.pages.flatMap((page) =>
    page.status === 200 ? [page.data.total] : [],
  );
  if (!totals.length) return true;
  if (totals.some((pageTotal) => pageTotal !== total)) return false;
  const loaded = flattenSessions(pages);
  return preview.every((session, index) => loaded[index]?.id === session.id);
}

function newestFirst(a: SessionSummaryResponse, b: SessionSummaryResponse) {
  const byTime =
    new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime();
  if (byTime) return byTime;
  if (a.id === b.id) return 0;
  return a.id < b.id ? 1 : -1;
}

// Collapsing a group is a per-device preference, kept as the keys of the
// groups the user collapsed. Every group starts open.
export function isGroupCollapsed(expertId: string | null) {
  return readCollapsedGroups().includes(expertId ?? OTTO_GROUP_KEY);
}

export function setGroupCollapsed(expertId: string | null, collapsed: boolean) {
  const key = expertId ?? OTTO_GROUP_KEY;
  const others = readCollapsedGroups().filter((group) => group !== key);
  storage.set(
    Key.RECENT_CHATS_COLLAPSED_GROUPS,
    JSON.stringify(collapsed ? [...others, key] : others),
  );
}

function readCollapsedGroups(): string[] {
  try {
    const parsed: unknown = JSON.parse(
      storage.get(Key.RECENT_CHATS_COLLAPSED_GROUPS) ?? "[]",
    );
    return Array.isArray(parsed)
      ? parsed.filter((key): key is string => typeof key === "string")
      : [];
  } catch {
    return [];
  }
}
