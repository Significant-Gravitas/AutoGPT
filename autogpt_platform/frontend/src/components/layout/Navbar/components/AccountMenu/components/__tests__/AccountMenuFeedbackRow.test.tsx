import { act, render, screen } from "@/tests/integrations/test-utils";
import { describe, expect, test, vi } from "vitest";
import { AccountMenuFeedbackRow } from "../AccountMenuFeedbackRow";

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: vi
    .fn()
    .mockResolvedValue({ user: { email: "user@example.com" } }),
}));

vi.mock("@sentry/nextjs", () => ({
  getReplay: vi.fn(() => ({ getReplayId: () => "replay-123" })),
}));

describe("AccountMenuFeedbackRow", () => {
  test("renders a Give feedback button wired to the Tally feedback form", async () => {
    render(<AccountMenuFeedbackRow />);
    await act(async () => {});

    const button = screen.getByRole("button", { name: "Give feedback" });

    expect(button.getAttribute("data-tally-open")).toBe("3yx2L0");
    expect(button.getAttribute("data-tally-emoji-text")).toBe("👋");
    expect(button.getAttribute("data-tally-emoji-animation")).toBe("wave");
    expect(button.getAttribute("data-sentry-replay-id")).toBe("replay-123");
    expect(button.getAttribute("data-sentry-replay-url")).toBe(
      "https://significant-gravitas.sentry.io/replays/replay-123/",
    );
    expect(button.getAttribute("data-is-authenticated")).toBe("true");
  });
});
