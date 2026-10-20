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

  it("keeps only buttons that stay on the platform", () => {
    const notice = getScheduledChatNotice({
      ...NOTICE,
      actions: [
        { kind: "link", label: "Upgrade", href: "/settings/billing" },
        { kind: "link", label: "Elsewhere", href: "https://example.com" },
        { kind: "link", label: "Protocol-relative", href: "//example.com" },
        { kind: "link", label: "Backslash", href: "/\\example.com" },
        { kind: "link", label: "Script", href: "javascript:alert(1)" },
        { kind: "resume", label: "Resume" },
        "not an action",
      ],
    });

    expect(notice?.actions.map((action) => action.label)).toEqual(["Upgrade"]);
  });
});
