import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { SubscriptionTierSection } from "../SubscriptionTierSection";

// Mock next/navigation
const mockSearchParams = new URLSearchParams();
const mockRouterReplace = vi.fn();
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => mockSearchParams,
    useRouter: () => ({ push: vi.fn(), replace: mockRouterReplace }),
    usePathname: () => "/profile/credits",
  };
});

// Mock toast
const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: mockToast }),
}));

// Mock feature flags — default to payment enabled so button tests work
let mockPaymentEnabled = true;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ENABLE_PLATFORM_PAYMENT: "enable-platform-payment" },
  useGetFlag: () => mockPaymentEnabled,
}));

// Mock generated API hooks
const mockUseGetSubscriptionStatus = vi.fn();
const mockUseUpdateSubscriptionTier = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: unknown) =>
    mockUseGetSubscriptionStatus(opts),
  useUpdateSubscriptionTier: () => mockUseUpdateSubscriptionTier(),
}));

// Mock Dialog (Radix portals don't work in happy-dom)
const MockDialogContent = ({ children }: { children: React.ReactNode }) => (
  <div>{children}</div>
);
const MockDialogFooter = ({ children }: { children: React.ReactNode }) => (
  <div>{children}</div>
);
function MockDialog({
  controlled,
  children,
}: {
  controlled?: { isOpen: boolean; set: (open: boolean) => void };
  children: React.ReactNode;
  [key: string]: unknown;
}) {
  return controlled?.isOpen ? <div role="dialog">{children}</div> : null;
}
MockDialog.Content = MockDialogContent;
MockDialog.Footer = MockDialogFooter;
vi.mock("@/components/molecules/Dialog/Dialog", () => ({
  Dialog: MockDialog,
}));

function makeSubscription({
  tier = "FREE",
  monthlyCost = 0,
  tierCosts = { FREE: 0, PRO: 1999, BUSINESS: 4999, ENTERPRISE: 0 },
  prorationCreditCents = 0,
}: {
  tier?: string;
  monthlyCost?: number;
  tierCosts?: Record<string, number>;
  prorationCreditCents?: number;
} = {}) {
  return {
    tier,
    monthly_cost: monthlyCost,
    tier_costs: tierCosts,
    proration_credit_cents: prorationCreditCents,
  };
}

function setupMocks({
  subscription = makeSubscription(),
  isLoading = false,
  queryError = null as Error | null,
  mutateFn = vi.fn().mockResolvedValue({ status: 200, data: { url: "" } }),
  isPending = false,
  variables = undefined as { data?: { tier?: string } } | undefined,
} = {}) {
  // The hook uses select: (data) => (data.status === 200 ? data.data : null)
  // so the data value returned by the hook is already the transformed subscription object.
  // We simulate that by returning the subscription directly as data.
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: subscription,
    isLoading,
    error: queryError,
    refetch: vi.fn(),
  });
  mockUseUpdateSubscriptionTier.mockReturnValue({
    mutateAsync: mutateFn,
    isPending,
    variables,
  });
}

afterEach(() => {
  cleanup();
  mockUseGetSubscriptionStatus.mockReset();
  mockUseUpdateSubscriptionTier.mockReset();
  mockToast.mockReset();
  mockRouterReplace.mockReset();
  mockSearchParams.delete("subscription");
  mockPaymentEnabled = true;
});

describe("SubscriptionTierSection", () => {
  it("renders skeleton cards while loading", () => {
    setupMocks({ isLoading: true });
    render(<SubscriptionTierSection />);
    // Just verify we're rendering something (not null) and no tier cards
    expect(screen.queryByText("Pro")).toBeNull();
    expect(screen.queryByText("Business")).toBeNull();
  });

  it("renders error message when subscription fetch fails", () => {
    setupMocks({
      queryError: new Error("Network error"),
      subscription: makeSubscription(),
    });
    // Override the data to simulate failed state
    mockUseGetSubscriptionStatus.mockReturnValue({
      data: null,
      isLoading: false,
      error: new Error("Network error"),
      refetch: vi.fn(),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByRole("alert")).toBeDefined();
    expect(screen.getByText(/failed to load subscription info/i)).toBeDefined();
  });

  it("renders all three tier cards for FREE user", () => {
    setupMocks();
    render(<SubscriptionTierSection />);
    // Use getAllByText to account for the tier label AND cost display both containing "Free"
    expect(screen.getAllByText("Free").length).toBeGreaterThan(0);
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Business")).toBeDefined();
  });

  it("shows Current badge on the active tier", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("Current")).toBeDefined();
    // Upgrade to PRO button should NOT exist; Upgrade to BUSINESS and Downgrade to Free should
    expect(
      screen.queryByRole("button", { name: /upgrade to pro/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /upgrade to business/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /downgrade to free/i }),
    ).toBeDefined();
  });

  it("displays tier costs from the API", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "FREE",
        tierCosts: { FREE: 0, PRO: 1999, BUSINESS: 4999, ENTERPRISE: 0 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("$19.99/mo")).toBeDefined();
    expect(screen.getByText("$49.99/mo")).toBeDefined();
    // FREE tier label should still be visible (there may be multiple "Free" elements)
    expect(screen.getAllByText("Free").length).toBeGreaterThan(0);
  });

  it("shows 'Pricing available soon' when tier cost is 0 for a paid tier", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "FREE",
        tierCosts: { FREE: 0, PRO: 0, BUSINESS: 0, ENTERPRISE: 0 },
      }),
    });
    render(<SubscriptionTierSection />);
    // PRO and BUSINESS with cost=0 should show "Pricing available soon"
    expect(screen.getAllByText("Pricing available soon")).toHaveLength(2);
  });

  it("calls changeTier on upgrade click after confirmation dialog", async () => {
    const mutateFn = vi
      .fn()
      .mockResolvedValue({ status: 200, data: { url: "" } });
    setupMocks({ mutateFn });
    render(<SubscriptionTierSection />);

    // Clicking upgrade opens the confirmation dialog first
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));
    // Confirm via the dialog's "Continue to Checkout" button
    fireEvent.click(
      screen.getByRole("button", { name: /continue to checkout/i }),
    );

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({ tier: "PRO" }),
        }),
      );
    });
  });

  it("shows confirmation dialog on downgrade click", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);

    fireEvent.click(screen.getByRole("button", { name: /downgrade to free/i }));

    expect(screen.getByRole("dialog")).toBeDefined();
    // The dialog title text appears in both a div and a button — just check the dialog is open
    expect(screen.getAllByText(/confirm downgrade/i).length).toBeGreaterThan(0);
  });

  it("calls changeTier after downgrade confirmation", async () => {
    const mutateFn = vi
      .fn()
      .mockResolvedValue({ status: 200, data: { url: "" } });
    setupMocks({
      subscription: makeSubscription({ tier: "PRO" }),
      mutateFn,
    });
    render(<SubscriptionTierSection />);

    fireEvent.click(screen.getByRole("button", { name: /downgrade to free/i }));
    fireEvent.click(screen.getByRole("button", { name: /confirm downgrade/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({ tier: "FREE" }),
        }),
      );
    });
  });

  it("dismisses dialog when Cancel is clicked", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);

    fireEvent.click(screen.getByRole("button", { name: /downgrade to free/i }));
    expect(screen.getByRole("dialog")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
  });

  it("redirects to Stripe when checkout URL is returned", async () => {
    // Replace window.location with a plain object so assigning .href doesn't
    // trigger jsdom navigation (which would throw or reload the test page).
    const mockLocation = { href: "" };
    vi.stubGlobal("location", mockLocation);

    const mutateFn = vi.fn().mockResolvedValue({
      status: 200,
      data: { url: "https://checkout.stripe.com/pay/cs_test" },
    });
    setupMocks({ mutateFn });
    render(<SubscriptionTierSection />);

    // Upgrade opens confirmation dialog first — confirm via "Continue to Checkout"
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /continue to checkout/i }),
    );

    await waitFor(() => {
      expect(mockLocation.href).toBe("https://checkout.stripe.com/pay/cs_test");
    });

    vi.unstubAllGlobals();
  });

  it("shows an error alert when tier change fails", async () => {
    const mutateFn = vi.fn().mockRejectedValue(new Error("Stripe unavailable"));
    setupMocks({ mutateFn });
    render(<SubscriptionTierSection />);

    // Upgrade opens confirmation dialog first — confirm to trigger the mutation
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));
    fireEvent.click(
      screen.getByRole("button", { name: /continue to checkout/i }),
    );

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeDefined();
      expect(screen.getByText(/stripe unavailable/i)).toBeDefined();
    });
  });

  it("hides action buttons when payment flag is disabled", () => {
    mockPaymentEnabled = false;
    setupMocks({ subscription: makeSubscription({ tier: "FREE" }) });
    render(<SubscriptionTierSection />);
    // Tier cards still visible
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Business")).toBeDefined();
    // No upgrade/downgrade buttons
    expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /downgrade/i })).toBeNull();
  });

  it("shows ENTERPRISE message for ENTERPRISE tier users", () => {
    setupMocks({ subscription: makeSubscription({ tier: "ENTERPRISE" }) });
    render(<SubscriptionTierSection />);
    // Enterprise heading text appears in a <p> (may match multiple), just verify it exists
    expect(screen.getAllByText(/enterprise plan/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/managed by your administrator/i)).toBeDefined();
    // No standard tier cards should be rendered
    expect(screen.queryByText("Pro")).toBeNull();
    expect(screen.queryByText("Business")).toBeNull();
  });

  it("shows success toast and clears URL param when ?subscription=success is present", async () => {
    mockSearchParams.set("subscription", "success");
    setupMocks();
    render(<SubscriptionTierSection />);

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Subscription upgraded" }),
      );
    });
    // URL param must be stripped so a page refresh doesn't re-trigger the toast
    expect(mockRouterReplace).toHaveBeenCalledWith("/profile/credits");
  });

  it("clears URL param but shows no toast when ?subscription=cancelled is present", async () => {
    mockSearchParams.set("subscription", "cancelled");
    setupMocks();
    render(<SubscriptionTierSection />);

    // The cancelled param must be stripped from the URL (same hygiene as success)
    await waitFor(() => {
      expect(mockRouterReplace).toHaveBeenCalledWith("/profile/credits");
    });
    // No toast should fire — the user simply abandoned checkout
    expect(mockToast).not.toHaveBeenCalled();
  });
});
