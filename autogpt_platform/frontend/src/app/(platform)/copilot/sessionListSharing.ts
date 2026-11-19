import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { replaceEqualDeep } from "@tanstack/react-query";
import type { SessionGroupsResult } from "./sessionGroups";
import type { SessionListInfiniteData } from "./useSessionList";

/**
 * `structuralSharing` for the session list. The API client turns timestamps
 * into `Date`s, and TanStack's default sharing treats every `Date` as new, so
 * each refetch rebuilt every session object and every row redrew. This keeps
 * the cached object of each session whose fields are unchanged, matched by id
 * so a session that only moved (pinned, or bumped to the top) keeps its object
 * too, and returns the cached list itself when nothing changed at all.
 */
export function shareSessionListData(
  oldData: unknown,
  newData: unknown,
): unknown {
  const previous = oldData as SessionListInfiniteData | undefined;
  const next = newData as SessionListInfiniteData | undefined;
  if (!previous?.pages || !next?.pages) return newData;

  const cachedById = new Map<string, SessionSummaryResponse>();
  for (const page of previous.pages) {
    if (page.status !== 200) continue;
    for (const session of page.data.sessions) {
      cachedById.set(session.id, session);
    }
  }

  const pages = next.pages.map((page, index) => {
    if (page.status !== 200) return page;
    const sessions = page.data.sessions.map((session) => {
      const cached = cachedById.get(session.id);
      return cached && haveSameFields(cached, session) ? cached : session;
    });
    const cachedPage = previous.pages[index];
    // The response headers are never read, and a `Headers` object never
    // compares equal, so they alone must not make an unchanged page new.
    if (
      cachedPage?.status === 200 &&
      isSameList(cachedPage.data.sessions, sessions) &&
      haveSameFields(cachedPage.data, page.data, "sessions")
    ) {
      return cachedPage;
    }
    return { ...page, data: { ...page.data, sessions } };
  });
  const pageParams = replaceEqualDeep(previous.pageParams, next.pageParams);

  const unchanged =
    pageParams === previous.pageParams &&
    pages.length === previous.pages.length &&
    pages.every((page, index) => page === previous.pages[index]);
  return unchanged ? previous : { ...next, pages, pageParams };
}

/**
 * `structuralSharing` for the sidebar's grouped read, for the same reason as
 * {@link shareSessionListData}: each unchanged session, and each group whose
 * total and chats are unchanged, keeps its cached object, and an unchanged
 * read returns the cached result itself, so a poll that brings nothing new
 * redraws nothing.
 */
export function shareSessionGroupsData(
  oldData: unknown,
  newData: unknown,
): unknown {
  const previous = oldData as SessionGroupsResult | undefined;
  const next = newData as SessionGroupsResult | undefined;
  if (previous?.status !== 200 || next?.status !== 200) return newData;

  const cachedById = new Map<string, SessionSummaryResponse>();
  for (const session of previous.data.pinned ?? []) {
    cachedById.set(session.id, session);
  }
  for (const group of previous.data.groups) {
    for (const session of group.sessions) cachedById.set(session.id, session);
  }
  function share(sessions: SessionSummaryResponse[]) {
    return sessions.map((session) => {
      const cached = cachedById.get(session.id);
      return cached && haveSameFields(cached, session) ? cached : session;
    });
  }

  const cachedGroups = new Map(
    previous.data.groups.map((group) => [group.expert_id ?? null, group]),
  );
  const groups = next.data.groups.map((group) => {
    const sessions = share(group.sessions);
    const cached = cachedGroups.get(group.expert_id ?? null);
    return cached &&
      isSameList(cached.sessions, sessions) &&
      haveSameFields(cached, group, "sessions")
      ? cached
      : { ...group, sessions };
  });
  const previousPinned = previous.data.pinned ?? [];
  const sharedPinned = share(next.data.pinned ?? []);
  const pinned = isSameList(previousPinned, sharedPinned)
    ? previous.data.pinned
    : sharedPinned;

  // `groups` and `pinned` are the whole payload. The response headers are
  // never read, and a `Headers` object never compares equal, so they alone
  // must not make an unchanged read new.
  const unchanged =
    pinned === previous.data.pinned && isSameList(previous.data.groups, groups);
  if (unchanged) return previous;
  return { ...next, data: { ...next.data, groups, pinned } };
}

function isSameList(a: unknown[], b: unknown[]) {
  return a.length === b.length && a.every((item, i) => item === b[i]);
}

function haveSameFields<T extends object>(a: T, b: T, skip?: keyof T) {
  const aKeys = Object.keys(a) as (keyof T)[];
  if (aKeys.length !== Object.keys(b).length) return false;
  return aKeys.every(
    (key) => key === skip || (key in b && isSameValue(a[key], b[key])),
  );
}

function isSameValue(a: unknown, b: unknown) {
  if (Object.is(a, b)) return true;
  if (a instanceof Date && b instanceof Date) {
    return a.getTime() === b.getTime();
  }
  // Any other object counts as changed: a needless row render, never a
  // stale one.
  return false;
}
