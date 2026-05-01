import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock next/navigation — usePathname isn't available in jsdom.
let mockPathname = "/build";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({ replace: vi.fn() }),
}));

// Mock the LD flag hook — toggle paid-cohort vs beta-cohort per test.
let mockIsPaymentEnabled: boolean | undefined = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ENABLE_PLATFORM_PAYMENT: "enable-platform-payment" },
  useGetFlag: () => mockIsPaymentEnabled,
}));

// Mock the generated subscription-status hook — control isLoading + tier per test.
type MockSubscription = { tier: string };
let mockSubscriptionResult: {
  data: MockSubscription | null | undefined;
  isLoading: boolean;
} = { data: null, isLoading: false };
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: () => mockSubscriptionResult,
}));

// Mock PaywallModal — actual rendering depends on Radix portals + the full
// SubscriptionTierSection, neither of which behave well under jsdom. Stand-in
// placeholder lets us assert the gate-vs-no-gate decision without booting the
// real modal.
vi.mock("../PaywallModal", () => ({
  PaywallModal: () => <div data-testid="paywall-modal">paywall</div>,
}));

import { PaywallGate } from "../PaywallGate";

describe("PaywallGate", () => {
  beforeEach(() => {
    mockPathname = "/build";
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: null, isLoading: false };
  });

  it("renders children without modal when ENABLE_PLATFORM_PAYMENT flag is off (beta cohort)", () => {
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
  });

  it("renders modal over children when paid cohort + tier is NO_TIER", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.getByTestId("paywall-modal")).toBeDefined();
  });

  it("does not render modal when paid cohort + tier is PRO/MAX/BUSINESS", () => {
    mockIsPaymentEnabled = true;
    for (const tier of ["PRO", "MAX", "BUSINESS"]) {
      mockSubscriptionResult = { data: { tier }, isLoading: false };
      const { unmount } = render(
        <PaywallGate>
          <div>protected</div>
        </PaywallGate>,
      );
      expect(screen.queryByTestId("paywall-modal")).toBeNull();
      unmount();
    }
  });

  it("does not render modal on exempt routes (/profile, /admin, /auth, /login, etc.)", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    for (const path of [
      "/profile/credits",
      "/profile/account",
      "/admin/users",
      "/auth/callback",
      "/login",
      "/signup",
      "/reset-password",
      "/error",
      "/unauthorized",
      "/health",
    ]) {
      mockPathname = path;
      const { unmount } = render(
        <PaywallGate>
          <div>protected</div>
        </PaywallGate>,
      );
      expect(screen.queryByTestId("paywall-modal")).toBeNull();
      unmount();
    }
  });

  it("does not render modal while subscription status is still loading", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: undefined, isLoading: true };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
  });
});
