import { describe, expect, it, vi } from "vitest";

import { fireEvent, render, screen } from "@/tests/integrations/test-utils";

import { PendingChangeBanner } from "../PendingChangeBanner";

describe("PendingChangeBanner", () => {
  const baseProps = {
    currentTier: "PRO",
    pendingTier: "BASIC",
    // Use noon UTC so the formatted local date lands on the same day
    // regardless of the host timezone (important for CI runners).
    pendingEffectiveAt: "2026-05-01T12:00:00Z",
    onKeepCurrent: () => {},
    isBusy: false,
  };

  it("renders nothing when there is no effective date", () => {
    // Backend invariant: effective date is always set when pending_tier is —
    // rendering without it would produce a sentence with a missing noun, so
    // the component must bail rather than show "Scheduled ... on undefined".
    const { container } = render(
      <PendingChangeBanner {...baseProps} pendingEffectiveAt={null} />,
    );
    expect(container.firstChild).toBeNull();
  });

  it("shows cancellation copy when pending tier is BASIC", () => {
    render(<PendingChangeBanner {...baseProps} />);
    expect(screen.getByText(/cancel your subscription on/i)).toBeDefined();
    expect(screen.getByText("May 1, 2026")).toBeDefined();
    // Button reflects the CURRENT tier, not the pending one.
    expect(screen.getByRole("button", { name: /keep pro/i })).toBeDefined();
  });

  it("shows downgrade copy when pending tier is a paid tier", () => {
    render(
      <PendingChangeBanner
        {...baseProps}
        currentTier="MAX"
        pendingTier="PRO"
      />,
    );
    expect(screen.getByText(/downgrade to/i)).toBeDefined();
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByRole("button", { name: /keep max/i })).toBeDefined();
  });

  it("invokes onKeepCurrent when the button is clicked", () => {
    const onKeepCurrent = vi.fn();
    render(
      <PendingChangeBanner {...baseProps} onKeepCurrent={onKeepCurrent} />,
    );
    fireEvent.click(screen.getByRole("button", { name: /keep pro/i }));
    expect(onKeepCurrent).toHaveBeenCalledTimes(1);
  });

  it("disables the button and swaps the label while busy", () => {
    render(<PendingChangeBanner {...baseProps} isBusy />);
    const button = screen.getByRole("button", { name: /cancelling/i });
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });
});
