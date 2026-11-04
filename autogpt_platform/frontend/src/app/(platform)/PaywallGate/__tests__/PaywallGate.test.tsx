import { act, render, screen } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";

// Mock next/navigation — usePathname isn't available in jsdom.
let mockPathname = "/build";
vi.mock("next/navigation", () => ({
  usePathname: () => mockPathname,
  useRouter: () => ({ replace: vi.fn() }),
}));

// Mock the auth hook — toggle logged-in vs logged-out per test.
let mockIsLoggedIn = true;
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: mockIsLoggedIn }),
}));

// Toggle the deployment's payment setting per test.
let mockIsPaymentEnabled = false;
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({
    isPaymentEnabled: mockIsPaymentEnabled,
    isResolved: true,
  }),
}));

// Mock the generated subscription-status hook — control isLoading + tier per test.
type MockSubscription = { tier: string };
let mockSubscriptionResult: {
  data: MockSubscription | null | undefined;
  isLoading: boolean;
} = { data: null, isLoading: false };
let subscriptionQueryEnabled: boolean | undefined;
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (options: { query?: { enabled?: boolean } }) => {
    subscriptionQueryEnabled = options.query?.enabled;
    return mockSubscriptionResult;
  },
}));

// Mock environment — default to cloud so the gate logic is exercised; flip to
// local per test to assert the local-dev bypass.
let mockIsLocal = false;
vi.mock("@/services/environment", () => ({
  environment: { isLocal: () => mockIsLocal, isServerSide: () => false },
}));

// Mock PaywallModal — actual rendering depends on Radix portals + the full
// SubscriptionTierSection, neither of which behave well under jsdom. Stand-in
// placeholder lets us assert the gate-vs-no-gate decision without booting the
// real modal.
vi.mock("../PaywallModal", () => ({
  PaywallModal: () => <div data-testid="paywall-modal">paywall</div>,
}));

vi.mock("../../components/WorkflowsMovedNotice/WorkflowsMovedNotice", () => ({
  WorkflowsMovedNotice: () => (
    <div data-testid="migration-notice">migration</div>
  ),
}));

import { useOrgTeamStore } from "@/services/org-team/store";
import { PaywallGate } from "../PaywallGate";

const PERSONAL_ORG = {
  id: "org-personal",
  name: "Jane's Org",
  slug: "jane",
  avatarUrl: null,
  isPersonal: true,
  memberCount: 1,
};
const TEAM_ORG = {
  ...PERSONAL_ORG,
  id: "org-team",
  name: "Acme Inc",
  slug: "acme",
  isPersonal: false,
  memberCount: 4,
};

function workIn(org: typeof PERSONAL_ORG | null, isLoaded = true) {
  useOrgTeamStore.setState({
    activeOrgID: org?.id ?? null,
    orgs: org ? [PERSONAL_ORG, TEAM_ORG] : [],
    isLoaded,
  });
}

describe("PaywallGate", () => {
  beforeEach(() => {
    mockPathname = "/build";
    mockIsLoggedIn = true;
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: null, isLoading: false };
    subscriptionQueryEnabled = undefined;
    mockIsLocal = false;
    workIn(PERSONAL_ORG);
  });

  it("renders children without modal when payment is off", () => {
    mockIsPaymentEnabled = false;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
    expect(screen.getByTestId("migration-notice")).toBeDefined();
  });

  it("renders modal over children when payment is on + tier is NO_TIER", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.getByTestId("paywall-modal")).toBeDefined();
    expect(screen.queryByTestId("migration-notice")).toBeNull();
  });

  it("does not render modal once the user is logged out, even when payment is on + tier is NO_TIER", () => {
    mockIsPaymentEnabled = true;
    mockIsLoggedIn = false;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
    expect(subscriptionQueryEnabled).toBe(false);
  });

  it("does not render modal in local dev even when payment is on + tier is NO_TIER", () => {
    mockIsPaymentEnabled = true;
    mockIsLocal = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
  });

  it("does not render modal when payment is on + tier is PRO/MAX/BUSINESS", () => {
    mockIsPaymentEnabled = true;
    for (const tier of ["PRO", "MAX", "BUSINESS"]) {
      mockSubscriptionResult = { data: { tier }, isLoading: false };
      const { unmount } = render(
        <PaywallGate>
          <div>protected</div>
        </PaywallGate>,
      );
      expect(screen.queryByTestId("paywall-modal")).toBeNull();
      expect(screen.getByTestId("migration-notice")).toBeDefined();
      unmount();
    }
  });

  it("does not render modal on exempt routes (/profile, /admin, /auth, /login, etc.)", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    for (const path of [
      "/profile/credits",
      "/profile/account",
      "/settings/billing",
      "/settings/profile",
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
    expect(screen.queryByTestId("migration-notice")).toBeNull();
  });

  it("never opens the personal plan picker in a team organization, whose own plan decides there", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    workIn(TEAM_ORG);
    render(
      <PaywallGate>
        <div>protected</div>
      </PaywallGate>,
    );
    expect(screen.getByText("protected")).toBeDefined();
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
    expect(screen.getByTestId("migration-notice")).toBeDefined();
  });

  it("waits for the organizations, so a team organization's member never sees the picker flash", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    workIn(null, false);
    render(<PaywallGate>protected</PaywallGate>);
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
    expect(screen.queryByTestId("migration-notice")).toBeNull();

    act(() => workIn(PERSONAL_ORG));
    expect(screen.getByTestId("paywall-modal")).toBeDefined();
  });

  it("falls back to the person's own plan when the organizations can't load", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    workIn(null, true);
    render(<PaywallGate>protected</PaywallGate>);
    expect(screen.getByTestId("paywall-modal")).toBeDefined();
  });

  it("defers the migration notice until the paywall clears", () => {
    mockIsPaymentEnabled = true;
    mockSubscriptionResult = { data: undefined, isLoading: true };
    const { rerender } = render(<PaywallGate>protected</PaywallGate>);
    expect(screen.queryByTestId("migration-notice")).toBeNull();

    mockSubscriptionResult = { data: { tier: "NO_TIER" }, isLoading: false };
    rerender(<PaywallGate>protected</PaywallGate>);
    expect(screen.getByTestId("paywall-modal")).toBeDefined();
    expect(screen.queryByTestId("migration-notice")).toBeNull();

    mockSubscriptionResult = { data: { tier: "PRO" }, isLoading: false };
    rerender(<PaywallGate>protected</PaywallGate>);
    expect(screen.queryByTestId("paywall-modal")).toBeNull();
    expect(screen.getByTestId("migration-notice")).toBeDefined();
  });
});
