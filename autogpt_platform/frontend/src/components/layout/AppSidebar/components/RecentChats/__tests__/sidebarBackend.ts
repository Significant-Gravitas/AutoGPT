import { http, HttpResponse } from "msw";

/** A chat as the sidebar tests describe it; everything else is filled in. */
export interface SidebarChat {
  id: string;
  title: string;
  expert_id?: string | null;
  is_processing?: boolean;
  is_pinned?: boolean;
  attention?: "needs_you" | "error" | "unread" | null;
  /** A turn the user hasn't seen. Implied by an unread or error marker; a
   *  chat that needs the user can have one too, as on the server. */
  unread?: boolean;
}

function isUnread(chat: SidebarChat) {
  return (
    chat.unread ?? (chat.attention === "unread" || chat.attention === "error")
  );
}

export interface SidebarRequest {
  path: "session-groups" | "sessions" | "sessions/read";
  params: URLSearchParams;
  body?: unknown;
}

// One minute apart, newest first, `Z`-suffixed so the API client turns them
// into `Date`s as it does in the app.
const NEWEST = Date.parse("2026-06-30T10:00:00Z");

/**
 * Serves the sidebar the way the backend does, from one list of chats given
 * newest first: the grouped read (each expert's real total and newest
 * `per_group` chats, the most recently active group first, pinned chats apart
 * when asked) and the plain list paging through one group. `chats` is read on
 * every request, so a test can change what the server holds between them.
 */
export function sidebarBackend(
  chats: () => SidebarChat[],
  requests: SidebarRequest[] = [],
) {
  return [
    http.get("*/api/chat/session-groups", ({ request }) => {
      const params = new URL(request.url).searchParams;
      requests.push({ path: "session-groups", params });
      return HttpResponse.json(groupedRead(chats(), params));
    }),
    http.get("*/api/chat/sessions", ({ request }) => {
      const params = new URL(request.url).searchParams;
      requests.push({ path: "sessions", params });
      return HttpResponse.json(listRead(chats(), params));
    }),
    // Marks the chats asked for read, as the server does: their unread and
    // error markers clear, and a chat that needs the user keeps its marker.
    http.post("*/api/chat/sessions/read", async ({ request }) => {
      const body = (await request.json()) as {
        expert_id?: string | null;
        autopilot_only?: boolean;
        pinned?: boolean | null;
      };
      requests.push({
        path: "sessions/read",
        params: new URL(request.url).searchParams,
        body,
      });
      let marked = 0;
      for (const chat of chats()) {
        const inScope =
          (body.expert_id
            ? chat.expert_id === body.expert_id
            : !body.autopilot_only || !chat.expert_id) &&
          (body.pinned == null || !!chat.is_pinned === body.pinned);
        if (inScope && isUnread(chat)) {
          chat.unread = false;
          if (chat.attention !== "needs_you") chat.attention = null;
          marked += 1;
        }
      }
      return HttpResponse.json({ marked });
    }),
  ];
}

export function toSummary(chat: SidebarChat, index: number) {
  const at = new Date(NEWEST - index * 60_000).toISOString();
  return {
    id: chat.id,
    title: chat.title,
    expert_id: chat.expert_id ?? null,
    is_processing: chat.is_processing ?? false,
    is_pinned: chat.is_pinned ?? false,
    attention: chat.attention ?? null,
    created_at: at,
    updated_at: at,
  };
}

/** The grouped read's body, for a test that answers it itself: each group's
 *  `min_per_group` newest chats, then the newest of the rest up to `limit`
 *  chats in all, as the backend picks them. */
export function groupedRead(chats: SidebarChat[], params: URLSearchParams) {
  const minPerGroup = Number(params.get("min_per_group") ?? "3");
  const limit = Number(params.get("limit") ?? "30");
  const pinnedApart = params.get("pinned_separately") === "true";
  const summaries = chats.map(toSummary);
  const grouped = pinnedApart
    ? summaries.filter((chat) => !chat.is_pinned)
    : summaries;
  const kept = keptIds(grouped, minPerGroup, limit);
  const byExpert = new Map<string | null, typeof summaries>();
  for (const chat of grouped) {
    byExpert.set(chat.expert_id, [
      ...(byExpert.get(chat.expert_id) ?? []),
      chat,
    ]);
  }
  return {
    groups: [...byExpert.entries()].map(([expertId, groupChats]) => ({
      expert_id: expertId,
      total: groupChats.length,
      attention_count: groupChats.filter((chat) => chat.attention).length,
      unread_count: groupChats.filter((summary) =>
        chats.some((chat) => chat.id === summary.id && isUnread(chat)),
      ).length,
      sessions: groupChats.filter((chat) => kept.has(chat.id)),
    })),
    pinned: pinnedApart ? summaries.filter((chat) => chat.is_pinned) : [],
  };
}

function keptIds(
  newestFirst: { id: string; expert_id: string | null }[],
  minPerGroup: number,
  limit: number,
) {
  const taken = new Map<string | null, number>();
  const guaranteed: string[] = [];
  const rest: string[] = [];
  for (const chat of newestFirst) {
    const count = taken.get(chat.expert_id) ?? 0;
    if (count < minPerGroup) {
      taken.set(chat.expert_id, count + 1);
      guaranteed.push(chat.id);
    } else {
      rest.push(chat.id);
    }
  }
  return new Set([
    ...guaranteed,
    ...rest.slice(0, Math.max(0, limit - guaranteed.length)),
  ]);
}

function listRead(chats: SidebarChat[], params: URLSearchParams) {
  const expertId = params.get("expert_id");
  const pinned = params.get("pinned");
  const offset = Number(params.get("offset") ?? "0");
  const limit = Number(params.get("limit") ?? "50");
  const rows = chats.map(toSummary).filter((chat) => {
    if (expertId && chat.expert_id !== expertId) return false;
    if (params.get("autopilot_only") === "true" && chat.expert_id) {
      return false;
    }
    if (pinned === "true" && !chat.is_pinned) return false;
    if (pinned === "false" && chat.is_pinned) return false;
    return true;
  });
  return { sessions: rows.slice(offset, offset + limit), total: rows.length };
}

/** `count` chats for one group, newest first: `${prefix} chat 1` onwards. */
export function groupChats(count: number, expertId?: string): SidebarChat[] {
  const prefix = expertId ?? "autopilot";
  return Array.from({ length: count }, (_, index) => ({
    id: `${prefix}-${index + 1}`,
    title: `${prefix} chat ${index + 1}`,
    expert_id: expertId ?? null,
  }));
}
