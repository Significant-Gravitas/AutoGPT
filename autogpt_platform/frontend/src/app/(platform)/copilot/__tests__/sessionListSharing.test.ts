import { describe, expect, it } from "vitest";
import { shareSessionListData } from "../sessionListSharing";
import type { SessionListInfiniteData } from "../useSessionList";

// The API client hands these over with `Date`s for timestamps, so the tests
// build them the same way.
function session(id: string, overrides: Record<string, unknown> = {}) {
  return {
    id,
    title: `Chat ${id}`,
    is_processing: false,
    is_pinned: false,
    created_at: new Date("2026-01-05T09:00:00Z"),
    updated_at: new Date("2026-01-05T09:00:00Z"),
    ...overrides,
  };
}

function listOf(...pages: ReturnType<typeof session>[][]) {
  return {
    pages: pages.map((sessions) => ({
      status: 200,
      data: { sessions, total: 3 },
      headers: new Headers(),
    })),
    pageParams: pages.map((_, i) => i * 50),
  } as unknown as SessionListInfiniteData;
}

describe("shareSessionListData", () => {
  it("returns the cached list when a refetch brings back the same sessions", () => {
    const cached = listOf([session("a"), session("b"), session("c")]);
    const refetched = listOf([session("a"), session("b"), session("c")]);

    expect(shareSessionListData(cached, refetched)).toBe(cached);
  });

  it("gives a new object only to the session that changed", () => {
    const cached = listOf([session("a"), session("b"), session("c")]);
    const refetched = listOf([
      session("a"),
      session("b", { title: "Renamed" }),
      session("c"),
    ]);

    const shared = shareSessionListData(
      cached,
      refetched,
    ) as SessionListInfiniteData;
    const [before, after] = [cached, shared].map((list) => {
      const page = list.pages[0];
      if (page.status !== 200) throw new Error("expected a 200 page");
      return page.data.sessions;
    });

    expect(after[0]).toBe(before[0]);
    expect(after[1]).not.toBe(before[1]);
    expect(after[1].title).toBe("Renamed");
    expect(after[2]).toBe(before[2]);
  });

  it("keeps the objects of sessions that only moved", () => {
    const cached = listOf([session("a"), session("b"), session("c")]);
    const refetched = listOf([
      session("c", { is_pinned: true }),
      session("a"),
      session("b"),
    ]);

    const shared = shareSessionListData(
      cached,
      refetched,
    ) as SessionListInfiniteData;
    const page = shared.pages[0];
    const cachedPage = cached.pages[0];
    if (page.status !== 200 || cachedPage.status !== 200) {
      throw new Error("expected 200 pages");
    }

    expect(page.data.sessions.map((s) => s.id)).toEqual(["c", "a", "b"]);
    expect(page.data.sessions[0]).not.toBe(cachedPage.data.sessions[2]);
    expect(page.data.sessions[1]).toBe(cachedPage.data.sessions[0]);
    expect(page.data.sessions[2]).toBe(cachedPage.data.sessions[1]);
  });

  it("treats a moved timestamp as a change", () => {
    const cached = listOf([session("a")]);
    const refetched = listOf([
      session("a", { updated_at: new Date("2026-01-06T09:00:00Z") }),
    ]);

    expect(shareSessionListData(cached, refetched)).not.toBe(cached);
  });

  it("keeps an unchanged older page when a new page is appended", () => {
    const cached = listOf([session("a")]);
    const withNext = listOf([session("a")], [session("b")]);

    const shared = shareSessionListData(
      cached,
      withNext,
    ) as SessionListInfiniteData;

    expect(shared.pages).toHaveLength(2);
    expect(shared.pages[0]).toBe(cached.pages[0]);
  });

  it("passes the first load through untouched", () => {
    const first = listOf([session("a")]);

    expect(shareSessionListData(undefined, first)).toBe(first);
  });
});
