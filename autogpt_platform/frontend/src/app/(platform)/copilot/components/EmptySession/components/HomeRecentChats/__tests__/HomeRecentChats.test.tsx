import { render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { HomeRecentChats } from "../HomeRecentChats";

function session(id: string, title: string, extra: object = {}) {
  return {
    id,
    title,
    created_at: "2026-11-03T10:00:00Z",
    updated_at: "2026-11-03T10:00:00Z",
    chat_status: "idle",
    is_processing: false,
    expert_id: null,
    attention: null,
    ...extra,
  };
}

vi.mock("@/app/(platform)/copilot/useSessionList", () => ({
  useSessionList: () => ({
    sessions: [
      session("a", "Weekly report", { is_processing: true }),
      session("b", "Pick a venue", { attention: "needs_you" }),
      session("c", "Sync the CRM", { attention: "failed" }),
      session("d", "Plan the offsite"),
    ],
  }),
}));

describe("HomeRecentChats", () => {
  test("each chat shows whether it is working, needs you, failed or is done", () => {
    render(<HomeRecentChats expertId={null} />);

    const names = screen
      .getAllByRole("link")
      .map((link) => link.textContent?.replace(/\s+/g, " "));
    expect(names).toEqual([
      expect.stringContaining("Working: Weekly report"),
      expect.stringContaining("Needs you: Pick a venue"),
      expect.stringContaining("Failed: Sync the CRM"),
      expect.stringContaining("Done: Plan the offsite"),
    ]);
  });
});
