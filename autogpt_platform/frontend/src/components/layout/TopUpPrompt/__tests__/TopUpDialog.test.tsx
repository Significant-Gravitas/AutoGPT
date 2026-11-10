import {
  getPatchV2UpdateFeedbackPromptMockHandler200,
  getPostV2ClaimFeedbackPromptMockHandler200,
} from "@/app/api/__generated__/endpoints/feedback-prompts/feedback-prompts.msw";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";

import { TopUpDialog } from "../TopUpDialog/TopUpDialog";

vi.mock("../TopUpForm/TopUpForm", () => ({
  TopUpForm: ({ submitLabel }: { submitLabel?: string }) => (
    <button type="submit">{submitLabel}</button>
  ),
}));

describe("TopUpDialog", () => {
  it("defaults to the out-of-credits copy", () => {
    render(<TopUpDialog isOpen onClose={() => {}} />);

    expect(screen.getByText("You're out of automation credits")).toBeDefined();
    expect(
      screen.getByText(/Top up to keep your workflows and experts running/),
    ).toBeDefined();
    expect(screen.queryByText("Add automation credits")).toBeNull();
  });

  it("uses the add-credits copy when opened deliberately from the wallet", () => {
    render(<TopUpDialog isOpen onClose={() => {}} variant="add-credits" />);

    expect(screen.getByText("Add automation credits")).toBeDefined();
    expect(
      screen.getByText(/Credits are used to run your workflows and experts/),
    ).toBeDefined();
    expect(screen.queryByText("You're out of automation credits")).toBeNull();
  });

  it("links to billing settings in both variants", () => {
    const { unmount } = render(<TopUpDialog isOpen onClose={() => {}} />);

    expect(
      screen
        .getByRole("link", { name: /enable auto-refill/ })
        .getAttribute("href"),
    ).toBe("/settings/billing");

    unmount();
    render(<TopUpDialog isOpen onClose={() => {}} variant="add-credits" />);

    expect(
      screen
        .getByRole("link", { name: /enable auto-refill/ })
        .getAttribute("href"),
    ).toBe("/settings/billing");
  });

  it("does not render its content while closed", () => {
    render(<TopUpDialog isOpen={false} onClose={() => {}} />);

    expect(screen.queryByText(/automation credits/i)).toBeNull();
  });

  it("calls onClose once when the close button is used", async () => {
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("notifies close exactly once on Escape", async () => {
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("has no Not now when opened from the wallet", () => {
    render(<TopUpDialog isOpen onClose={() => {}} variant="add-credits" />);

    expect(screen.queryByRole("button", { name: "Not now" })).toBeNull();
  });
});

describe("TopUpDialog — Not now", () => {
  function answerWith(claimID: string | null) {
    const patches: unknown[] = [];
    server.use(
      getPostV2ClaimFeedbackPromptMockHandler200({ id: claimID }),
      getPatchV2UpdateFeedbackPromptMockHandler200(async ({ request }) => {
        patches.push(await request.clone().json());
        return {
          id: "prompt-1",
          kind: "top_up",
          answered: true,
          dismissed: false,
          reason: null,
          score: null,
        };
      }),
    );
    return patches;
  }

  it("just closes when no question is due", async () => {
    answerWith(null);
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.click(screen.getByRole("button", { name: "Not now" }));

    await waitFor(() => expect(onClose).toHaveBeenCalledTimes(1));
    expect(
      screen.queryByText("What would make topping up worth it?"),
    ).toBeNull();
  });

  it("asks in the same dialog and saves the answer", async () => {
    const patches = answerWith("prompt-1");
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.click(screen.getByRole("button", { name: "Not now" }));
    await userEvent.click(
      await screen.findByRole("button", { name: "Smaller amounts" }),
    );
    await userEvent.click(screen.getByRole("button", { name: "Send" }));

    expect(await screen.findByText(/we read every one/)).toBeDefined();
    expect(screen.getAllByRole("dialog")).toHaveLength(1);
    await waitFor(() =>
      expect(patches).toEqual([
        { reason: "smaller_amounts" },
        { reason: "smaller_amounts" },
      ]),
    );
    expect(onClose).not.toHaveBeenCalled();
  });

  it("records closing the question unanswered", async () => {
    const patches = answerWith("prompt-1");
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.click(screen.getByRole("button", { name: "Not now" }));
    await screen.findByText("What would make topping up worth it?");
    await userEvent.click(screen.getByRole("button", { name: "Close" }));

    await waitFor(() => expect(patches).toEqual([{ dismissed: true }]));
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});
