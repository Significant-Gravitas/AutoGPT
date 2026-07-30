import { render, screen } from "@/tests/integrations/test-utils";
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
    expect(screen.getByText(/Top up to keep your agents/)).toBeDefined();
    expect(screen.queryByText("Add automation credits")).toBeNull();
  });

  it("uses the add-credits copy when opened deliberately from the wallet", () => {
    render(<TopUpDialog isOpen onClose={() => {}} variant="add-credits" />);

    expect(screen.getByText("Add automation credits")).toBeDefined();
    expect(
      screen.getByText(/Credits are used to run your agents/),
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

  // Escape is handled twice by the design-system Dialog: `DialogWrap` passes
  // `onEscapeKeyDown={handleClose}`, and Radix's dismissable layer then also
  // fires `onOpenChange(false)` on the root. Both land on `controlled.set`.
  // Closing is idempotent so nothing breaks, but pin the count — if the Dialog
  // is ever fixed this test should be the thing that notices.
  it("notifies close on Escape (twice, via the shared Dialog)", async () => {
    const onClose = vi.fn();
    render(<TopUpDialog isOpen onClose={onClose} />);

    await userEvent.keyboard("{Escape}");

    expect(onClose).toHaveBeenCalledTimes(2);
  });
});
