import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock next/navigation — usePathname/useRouter aren't available in jsdom.
const mockReplace = vi.fn();
let mockPathname = "/build";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({ replace: mockReplace }),
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

import { PaywallGate } from "../PaywallGate";

describe("PaywallGate", () => {
  beforeEach(() => {
    mockReplace.mockClear();
    mockPathname = "/build";
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: null, isLoading: false };
  });

  it("renders children without redirecting when ENABLE_PLATFORM_PAYMENT flag is off (beta cohort)", () => {
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: { tier: "BASIC" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(mockReplace).not.toHaveBeenCalled();
  });

  it("redirects to /profile/credits when paid cohort + tier is BASIC", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "BASIC" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(mockReplace).toHaveBeenCalledWith("/profile/credits");
  });

  it("does not redirect when paid cohort + tier is PRO/MAX/BUSINESS", () => {
    mockIsPaymentEnabled = true;
    for (const tier of ["PRO", "MAX", "BUSINESS"]) {
      mockReplace.mockClear();
      mockSubscriptionResult = { data: { tier }, isLoading: false };
      render(
        <PaywallGate>
          <div>protected</div>
        </PaywallGate>,
      );
      expect(mockReplace).not.toHaveBeenCalled();
    }
  });

  it("does not redirect on exempt routes (/profile, /admin, /auth, /login, etc.)", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "BASIC" }, isLoading: false };
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
      mockReplace.mockClear();
      mockPathname = path;
      render(
        <PaywallGate>
          <div>protected</div>
        </PaywallGate>,
      );
      expect(mockReplace).not.toHaveBeenCalled();
    }
  });

  it("does not redirect while subscription status is still loading", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: undefined, isLoading: true };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(mockReplace).not.toHaveBeenCalled();
  });
});
