import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  isPreExpertsUser,
  isWorkflowsMovedNoticeRoute,
  hasPendingChatHandoff,
  peekWorkflowsMovedNoticeSeen,
  setWorkflowsMovedNoticeSeen,
} from "../helpers";

beforeEach(() => {
  window.localStorage.clear();
  window.sessionStorage.clear();
  vi.restoreAllMocks();
});

describe("the experts release audience", () => {
  it.each([
    ["2026-09-20T23:59:59.999Z", true],
    ["2026-09-21T00:00:00Z", false],
    ["2026-09-22T12:00:00Z", false],
    ["2026-09-21T01:00:00+02:00", true],
    ["not a date", false],
    ["", false],
    [undefined, false],
  ])("checks signup %s against the confirmed release", (signup, expected) => {
    expect(isPreExpertsUser(signup)).toBe(expected);
  });
});

describe("safe landing routes", () => {
  it.each([
    "sessionId=chat-1",
    "expertId=expert-1",
    "kickoff=1",
    "autosubmit=true",
    "new=1",
    "modal=skills",
  ])("defers copilot with %s", (search) =>
    expect(
      isWorkflowsMovedNoticeRoute("/copilot", new URLSearchParams(search)),
    ).toBe(false),
  );

  it.each(["/copilot", "/team", "/library", "/team/"])(
    "allows %s",
    (pathname) => expect(isWorkflowsMovedNoticeRoute(pathname)).toBe(true),
  );

  it.each([
    "/copilot/session-1",
    "/team/expert-1",
    "/library/agents/agent-1",
    "/build",
    "/build/agent-1",
    "/settings",
    "/marketplace",
    "/onboarding",
    null,
  ])("does not interrupt %s", (pathname) => {
    expect(isWorkflowsMovedNoticeRoute(pathname)).toBe(false);
  });
});

describe("chat handoffs", () => {
  it.each(["A", "B"])("defers an unconsumed onboarding path %s", (path) => {
    window.sessionStorage.setItem("autogpt:onboarding-intro-path", path);

    expect(hasPendingChatHandoff()).toBe(true);
    expect(window.sessionStorage.getItem("autogpt:onboarding-intro-path")).toBe(
      path,
    );
  });

  it("defers imported prompts and welcome handoffs", () => {
    window.sessionStorage.setItem("importWorkflowPrompt", "Run my workflow");
    expect(hasPendingChatHandoff()).toBe(true);
    window.sessionStorage.removeItem("importWorkflowPrompt");
    window.sessionStorage.setItem("autogpt:onboarding-welcome-pending", "1");
    expect(hasPendingChatHandoff()).toBe(true);
    window.sessionStorage.clear();
  });

  it("defers safely when session storage is unavailable", () => {
    vi.spyOn(window.sessionStorage, "getItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    expect(hasPendingChatHandoff()).toBe(true);
  });
});

describe("per-account dismissal cache", () => {
  it("keeps both accounts dismissed when users switch back and forth", () => {
    setWorkflowsMovedNoticeSeen("user-1");
    expect(peekWorkflowsMovedNoticeSeen("user-2")).toBe(false);
    setWorkflowsMovedNoticeSeen("user-2");
    expect(peekWorkflowsMovedNoticeSeen("user-1")).toBe(true);
    expect(peekWorkflowsMovedNoticeSeen("user-2")).toBe(true);
  });

  it("ignores an unresolved account", () => {
    setWorkflowsMovedNoticeSeen(null);
    expect(peekWorkflowsMovedNoticeSeen(null)).toBe(false);
    expect(window.localStorage.length).toBe(0);
  });

  it("tolerates browsers that deny storage access", () => {
    vi.spyOn(window.localStorage, "getItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    vi.spyOn(window.localStorage, "setItem").mockImplementation(() => {
      throw new Error("Storage unavailable");
    });
    expect(peekWorkflowsMovedNoticeSeen("user-1")).toBe(false);
    expect(() => setWorkflowsMovedNoticeSeen("user-1")).not.toThrow();
  });
});
