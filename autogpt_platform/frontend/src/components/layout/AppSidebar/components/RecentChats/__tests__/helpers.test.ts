import { Key, storage } from "@/services/storage/local-storage";
import { afterEach, describe, expect, it } from "vitest";

import {
  getNewChatHref,
  isGroupCollapsed,
  mergeGroupSessions,
  setGroupCollapsed,
  withoutDeletingSessions,
} from "../helpers";
import { toSummary } from "./sidebarBackend";

function summaries(...ids: string[]) {
  return ids.map((id, index) => toSummary({ id, title: id }, index));
}

describe("getNewChatHref", () => {
  const active = new Set(["expert-maria", "expert/slash"]);

  it("sends the Otto group to the plain home page", () => {
    expect(getNewChatHref(null, active)).toBe("/home");
  });

  it("addresses an active expert and asks for a fresh thread", () => {
    expect(getNewChatHref("expert-maria", active)).toBe(
      "/home?expertId=expert-maria&new=1",
    );
  });

  it("encodes the expert id", () => {
    expect(getNewChatHref("expert/slash", active)).toBe(
      "/home?expertId=expert%2Fslash&new=1",
    );
  });

  it("gives fired experts no link at all", () => {
    expect(getNewChatHref("expert-max", active)).toBeNull();
  });
});

describe("mergeGroupSessions", () => {
  it("keeps the preview's copy of a chat that both reads hold", () => {
    const [fresh, second, third] = summaries("a", "b", "c");
    const stale = { ...fresh, title: "old title" };

    const merged = mergeGroupSessions([fresh], [stale, second, third]);

    expect(merged.map((session) => session.id)).toEqual(["a", "b", "c"]);
    expect(merged[0]).toBe(fresh);
  });

  it("moves a chat that got newer since the older pages were read to the top", () => {
    const [first, second] = summaries("a", "b");
    const bumped = {
      ...second,
      updated_at: new Date(Date.parse(first.updated_at) + 60_000).toISOString(),
    };

    expect(
      mergeGroupSessions([bumped], [first, second]).map((s) => s.id),
    ).toEqual(["b", "a"]);
  });

  it("hands back the preview itself before any older page is read", () => {
    const preview = summaries("a", "b");
    expect(mergeGroupSessions(preview, [])).toBe(preview);
  });
});

describe("withoutDeletingSessions", () => {
  const otto = { expert_id: null, total: 5, sessions: summaries("a", "b") };
  const maria = { expert_id: "maria", total: 1, sessions: summaries("m") };

  it("leaves every group's object alone when nothing is being deleted", () => {
    const groups = [otto, maria];
    expect(withoutDeletingSessions(groups, [])).toBe(groups);
  });

  it("lowers the total of the group losing a chat and drops an emptied one", () => {
    const [ottoAfter, ...rest] = withoutDeletingSessions(
      [otto, maria],
      ["a", "m"],
    );

    expect(ottoAfter.total).toBe(4);
    expect(ottoAfter.sessions.map((session) => session.id)).toEqual(["b"]);
    expect(rest).toEqual([]);
  });
});

describe("collapsed groups", () => {
  afterEach(() => {
    storage.clean(Key.RECENT_CHATS_COLLAPSED_GROUPS);
  });

  it("starts every group open and remembers each collapse on its own", () => {
    expect(isGroupCollapsed(null)).toBe(false);
    expect(isGroupCollapsed("maria")).toBe(false);

    setGroupCollapsed("maria", true);
    expect(isGroupCollapsed("maria")).toBe(true);
    expect(isGroupCollapsed(null)).toBe(false);

    setGroupCollapsed("maria", false);
    expect(isGroupCollapsed("maria")).toBe(false);
  });

  it("treats unreadable stored state as nothing collapsed", () => {
    storage.set(Key.RECENT_CHATS_COLLAPSED_GROUPS, "{not json");
    expect(isGroupCollapsed("maria")).toBe(false);
  });
});
