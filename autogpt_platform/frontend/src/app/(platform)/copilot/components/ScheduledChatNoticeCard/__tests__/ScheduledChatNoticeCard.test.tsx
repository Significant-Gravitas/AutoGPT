import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, it } from "vitest";
import { ScheduledChatNoticeCard } from "../ScheduledChatNoticeCard";
import type { ScheduledChatNotice } from "../helpers";

const NOTICE: ScheduledChatNotice = {
  reason: "usage",
  headline: "Skipped: daily usage limit reached",
  body: "“Briefing” didn't run. Scheduled chats run again when the limit resets at 00:00 UTC.",
  actions: [{ kind: "link", label: "Upgrade", href: "/settings/billing" }],
};

describe("ScheduledChatNoticeCard", () => {
  it("says why the run was skipped and links to the fix", () => {
    render(<ScheduledChatNoticeCard notice={NOTICE} />);

    expect(
      screen.getByText("Skipped: daily usage limit reached"),
    ).toBeDefined();
    expect(screen.getByText(/didn't run/)).toBeDefined();
    expect(
      screen.getByRole("link", { name: "Upgrade" }).getAttribute("href"),
    ).toBe("/settings/billing");
  });
});
