import { render } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

// Mock SubscriptionTierSection — its full render exercises the credits
// query, LD flag, and Stripe wiring which are out of scope for PaywallModal
// itself. The modal's job is to wrap the section in a non-dismissable Dialog;
// that's what we verify here.
vi.mock(
  "@/app/(platform)/profile/(user)/credits/components/SubscriptionTierSection/SubscriptionTierSection",
  () => ({
    SubscriptionTierSection: () => (
      <div data-testid="tier-section">tier picker</div>
    ),
  }),
);

// Mock the Dialog so we can introspect its props without booting Radix portals
// under jsdom (which doesn't reliably render them).
let lastDialogProps: Record<string, unknown> | null = null;
vi.mock("@/components/molecules/Dialog/Dialog", () => ({
  Dialog: Object.assign(
    function MockDialog(props: Record<string, unknown>) {
      lastDialogProps = props;
      return (
        <div data-testid="dialog">{props.children as React.ReactNode}</div>
      );
    },
    {
      Content: ({ children }: { children: React.ReactNode }) => (
        <div data-testid="dialog-content">{children}</div>
      ),
    },
  ),
}));

import { PaywallModal } from "../PaywallModal";

describe("PaywallModal", () => {
  it("renders SubscriptionTierSection inside a forceOpen, non-dismissable Dialog", () => {
    const { getByTestId } = render(<PaywallModal />);
    expect(getByTestId("dialog")).toBeDefined();
    expect(getByTestId("dialog-content")).toBeDefined();
    expect(getByTestId("tier-section")).toBeDefined();
    expect(lastDialogProps?.forceOpen).toBe(true);
    // controlled.set is a no-op (set: () => {}) so the modal can never be
    // closed by the Dialog's own state machinery — only by subscription state
    // changing in the consuming PaywallGate.
    expect(
      typeof (lastDialogProps?.controlled as { set?: unknown } | undefined)
        ?.set,
    ).toBe("function");
    expect(
      (lastDialogProps?.controlled as { isOpen?: boolean } | undefined)?.isOpen,
    ).toBe(true);
  });
});
