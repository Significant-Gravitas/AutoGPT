import { describe, expect, it } from "vitest";
import { getScheduledChatNotice } from "../helpers";

const NOTICE = {
  kind: "scheduled_chat_notice",
  hidden_from_model: true,
  reason: "usage",
  headline: "Skipped: daily usage limit reached",
  body: "“Briefing” didn't run.",
  actions: [{ kind: "link", label: "Upgrade", href: "/settings/billing" }],
};

describe("getScheduledChatNotice", () => {
  it("reads the note a skipped scheduled chat leaves", () => {
    expect(getScheduledChatNotice(NOTICE)).toEqual({
      reason: "usage",
      headline: "Skipped: daily usage limit reached",
      body: "“Briefing” didn't run.",
      scheduleKey: null,
      actions: [{ kind: "link", label: "Upgrade", href: "/settings/billing" }],
    });
  });

  it("ignores any other message", () => {
    expect(getScheduledChatNotice(undefined)).toBeNull();
    expect(getScheduledChatNotice({ kind: "expert_run" })).toBeNull();
    expect(
      getScheduledChatNotice({ kind: "scheduled_chat_notice", body: "x" }),
    ).toBeNull();
  });

  it("keeps only links that stay on the platform", () => {
    const notice = getScheduledChatNotice({
      ...NOTICE,
      actions: [
        { kind: "link", label: "Upgrade", href: "/settings/billing" },
        { kind: "link", label: "Elsewhere", href: "https://example.com" },
        { kind: "link", label: "Protocol-relative", href: "//example.com" },
        { kind: "link", label: "Backslash", href: "/\\example.com" },
        { kind: "link", label: "Script", href: "javascript:alert(1)" },
        "not an action",
      ],
    });

    expect(notice?.actions.map((action) => action.label)).toEqual(["Upgrade"]);
  });

  it("reads a paused chat's reconnect and resume", () => {
    const notice = getScheduledChatNotice({
      ...NOTICE,
      reason: "connection",
      schedule_key: "routine-1",
      actions: [
        { kind: "reconnect", label: "Reconnect Twitter", provider: "twitter" },
        { kind: "resume", label: "Resume" },
      ],
    });

    expect(notice?.scheduleKey).toBe("routine-1");
    expect(notice?.actions).toEqual([
      { kind: "reconnect", label: "Reconnect Twitter", provider: "twitter" },
      { kind: "resume", label: "Resume" },
    ]);
  });

  it("drops a resume it could not send anywhere", () => {
    const notice = getScheduledChatNotice({
      ...NOTICE,
      actions: [
        { kind: "resume", label: "Resume" },
        { kind: "reconnect", label: "Reconnect" },
        { kind: "run_less_often", label: "Run less often" },
      ],
    });

    expect(notice?.actions).toEqual([]);
  });

  it("reads a skipped chat's offer to run less often", () => {
    const notice = getScheduledChatNotice({
      ...NOTICE,
      schedule_key: "routine-1",
      actions: [
        { kind: "link", label: "Upgrade", href: "/settings/billing" },
        { kind: "run_less_often", label: "Run less often" },
      ],
    });

    expect(notice?.actions).toEqual([
      { kind: "link", label: "Upgrade", href: "/settings/billing" },
      { kind: "run_less_often", label: "Run less often" },
    ]);
  });
});
