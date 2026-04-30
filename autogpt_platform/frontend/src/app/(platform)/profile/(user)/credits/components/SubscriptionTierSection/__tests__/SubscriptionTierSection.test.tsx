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
  tier = "BASIC",
  monthlyCost = 0,
  tierCosts = { BASIC: 0, PRO: 1999, MAX: 32000, ENTERPRISE: 0 },
  tierMultipliers = { BASIC: 1, PRO: 5, MAX: 20, BUSINESS: 60 },
  prorationCreditCents = 0,
  pendingTier = null as string | null,
  pendingTierEffectiveAt = null as Date | string | null,
}: {
  tier?: string;
  monthlyCost?: number;
  tierCosts?: Record<string, number>;
  tierMultipliers?: Record<string, number>;
  prorationCreditCents?: number;
  pendingTier?: string | null;
  pendingTierEffectiveAt?: Date | string | null;
} = {}) {
  return {
    tier,
    monthly_cost: monthlyCost,
    tier_costs: tierCosts,
    tier_multipliers: tierMultipliers,
    proration_credit_cents: prorationCreditCents,
    pending_tier: pendingTier,
    pending_tier_effective_at: pendingTierEffectiveAt,
  };
}

function setupMocks({
  subscription = makeSubscription(),
  isLoading = false,
  queryError = null as Error | null,
  mutateFn = vi.fn().mockResolvedValue({ status: 200, data: { url: "" } }),
  isPending = false,
  variables = undefined as { data?: { tier?: string } } | undefined,
  refetchFn = vi.fn(),
} = {}) {
  // The hook uses select: (data) => (data.status === 200 ? data.data : null)
  // so the data value returned by the hook is already the transformed subscription object.
  // We simulate that by returning the subscription directly as data.
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: subscription,
    isLoading,
    error: queryError,
    refetch: refetchFn,
  });
  mockUseUpdateSubscriptionTier.mockReturnValue({
    mutateAsync: mutateFn,
    isPending,
    variables,
  });
  return { refetchFn, mutateFn };
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
    expect(screen.queryByText("Max")).toBeNull();
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

  it("renders all three tier cards for BASIC user", () => {
    setupMocks();
    render(<SubscriptionTierSection />);
    // BASIC tier card is labelled "Basic"; cost displays "Free" for BASIC@$0.
    expect(screen.getByText("Basic")).toBeDefined();
    expect(screen.getByText("Free")).toBeDefined();
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Max")).toBeDefined();
  });

  it("shows Current badge on the active tier", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("Current")).toBeDefined();
    // Upgrade to PRO button should NOT exist; Upgrade to Max and Downgrade to Basic should
    expect(
      screen.queryByRole("button", { name: /upgrade to pro/i }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: /upgrade to max/i }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    ).toBeDefined();
  });

  it("displays tier costs from the API", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0, PRO: 1999, MAX: 32000, ENTERPRISE: 0 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("$19.99/mo")).toBeDefined();
    expect(screen.getByText("$320.00/mo")).toBeDefined();
    // BASIC tier card label is "Basic"; its $0 cost renders "Free".
    expect(screen.getByText("Basic")).toBeDefined();
    expect(screen.getByText("Free")).toBeDefined();
  });

  it("shows 'Free' for any tier with cost = 0", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0, PRO: 0, MAX: 0, ENTERPRISE: 0 },
      }),
    });
    render(<SubscriptionTierSection />);
    // BASIC, PRO, MAX all with cost=0 should each render "Free".
    expect(screen.getAllByText("Free")).toHaveLength(3);
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

    fireEvent.click(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    );

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

    fireEvent.click(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    );
    fireEvent.click(screen.getByRole("button", { name: /confirm downgrade/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({ tier: "BASIC" }),
        }),
      );
    });
  });

  it("dismisses dialog when Cancel is clicked", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);

    fireEvent.click(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    );
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
    setupMocks({ subscription: makeSubscription({ tier: "BASIC" }) });
    render(<SubscriptionTierSection />);
    // Tier cards still visible
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Max")).toBeDefined();
    // No upgrade/downgrade buttons
    expect(screen.queryByRole("button", { name: /upgrade/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /downgrade/i })).toBeNull();
  });

  it("hides tiers that are missing from tier_costs (no LD price configured)", () => {
    // LD only has stripe-price-id-basic → only BASIC appears; PRO/Max/Business
    // cards must hide.
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("Basic")).toBeDefined();
    expect(screen.queryByText("Pro")).toBeNull();
    expect(screen.queryByText("Max")).toBeNull();
    expect(screen.queryByText("Business")).toBeNull();
  });

  it("renders Max card when tier_costs includes MAX and hides BUSINESS when its flag is unset", () => {
    // MAX is priced by default (stripe-price-id-max); BUSINESS stays reserved
    // (stripe-price-id-business unset) and must not render.
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0, PRO: 1999, MAX: 32000 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("Max")).toBeDefined();
    expect(screen.getByText("$320.00/mo")).toBeDefined();
    expect(screen.queryByText("Business")).toBeNull();
  });

  it("hides the current tier when its LD price is unset — no safety-net rendering", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        tierCosts: { PRO: 1999 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.queryByText("Max")).toBeNull();
    expect(screen.queryByText("Basic")).toBeNull();
  });

  it("renders rate-limit badges relative to the lowest visible tier", () => {
    // BASIC is baseline (1×) → no badge; PRO/MAX/BUSINESS show their ratio.
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0, PRO: 1999, MAX: 32000, BUSINESS: 14999 },
        tierMultipliers: { BASIC: 1, PRO: 5, MAX: 20, BUSINESS: 60 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.queryByText(/1\.0x rate limits/i)).toBeNull();
    expect(screen.getByText(/5\.0x rate limits/i)).toBeDefined();
    expect(screen.getByText(/20\.0x rate limits/i)).toBeDefined();
    expect(screen.getByText(/60\.0x rate limits/i)).toBeDefined();
  });

  it("rebases relative multipliers when the lowest tier is hidden", () => {
    // With BASIC hidden, PRO becomes the baseline (no badge) and MAX shows
    // "4.0x rate limits" (20 / 5).
    setupMocks({
      subscription: makeSubscription({
        tier: "PRO",
        tierCosts: { PRO: 1999, MAX: 32000 },
        tierMultipliers: { PRO: 5, MAX: 20 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.queryByText(/5\.0x rate limits/i)).toBeNull();
    expect(screen.getByText(/4\.0x rate limits/i)).toBeDefined();
  });

  it("honours fractional LD-provided multipliers in the relative display", () => {
    // LD can override the multiplier to a non-integer value (e.g. PRO=8.5×).
    setupMocks({
      subscription: makeSubscription({
        tier: "BASIC",
        tierCosts: { BASIC: 0, PRO: 1999 },
        tierMultipliers: { BASIC: 1, PRO: 8.5 },
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText(/8\.5x rate limits/i)).toBeDefined();
  });

  it("shows ENTERPRISE message for ENTERPRISE tier users", () => {
    setupMocks({ subscription: makeSubscription({ tier: "ENTERPRISE" }) });
    render(<SubscriptionTierSection />);
    // Enterprise heading text appears in a <p> (may match multiple), just verify it exists
    expect(screen.getAllByText(/enterprise plan/i).length).toBeGreaterThan(0);
    expect(screen.getByText(/managed by your administrator/i)).toBeDefined();
    // No standard tier cards should be rendered
    expect(screen.queryByText("Pro")).toBeNull();
    expect(screen.queryByText("Max")).toBeNull();
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

  it("renders pending-change banner when pending_tier is set", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.getByText(/scheduled to downgrade to/i)).toBeDefined();
    // Banner "Keep Max" button — the only Keep button, since the on-card
    // duplicate was removed in favour of the banner.
    expect(screen.getAllByRole("button", { name: /keep max/i })).toHaveLength(
      1,
    );
  });

  it("does not render pending-change banner when pending_tier is null", () => {
    setupMocks({
      subscription: makeSubscription({ tier: "MAX", pendingTier: null }),
    });
    render(<SubscriptionTierSection />);
    expect(screen.queryByText(/scheduled to downgrade/i)).toBeNull();
    expect(screen.queryByRole("button", { name: /keep max/i })).toBeNull();
  });

  it("clicking Keep [CurrentTier] in banner submits a same-tier update and refetches", async () => {
    // The cancel-pending route was collapsed into POST /credits/subscription as
    // a same-tier request. Clicking "Keep MAX" calls useUpdateSubscriptionTier
    // with tier === current tier so the backend releases any pending schedule.
    const mutateFn = vi
      .fn()
      .mockResolvedValue({ status: 200, data: { url: "", tier: "MAX" } });
    const refetchFn = vi.fn();
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
      mutateFn,
      refetchFn,
    });
    render(<SubscriptionTierSection />);

    fireEvent.click(screen.getByRole("button", { name: /keep max/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({ tier: "MAX" }),
        }),
      );
      expect(refetchFn).toHaveBeenCalled();
    });
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Pending subscription change cancelled.",
      }),
    );
  });

  it("uses end-of-period copy for paid→paid downgrade confirmation", () => {
    setupMocks({ subscription: makeSubscription({ tier: "MAX" }) });
    render(<SubscriptionTierSection />);

    fireEvent.click(screen.getByRole("button", { name: /downgrade to pro/i }));

    const dialog = screen.getByRole("dialog");
    expect(dialog.textContent).toMatch(
      /switching to pro takes effect at the end of your current billing period/i,
    );
    expect(dialog.textContent).toMatch(
      /you keep your current plan until then/i,
    );
    expect(dialog.textContent).toMatch(/no charge today/i);
    expect(dialog.textContent).not.toMatch(/take effect immediately/i);
  });

  it("shows destructive toast, tierError and still refetches when cancel-pending fails", async () => {
    // The catch branch inside cancelPendingChange is load-bearing: it surfaces
    // the error to the user AND re-issues a refetch so the UI reconciles if
    // the server actually succeeded (webhook delivered after our client-side
    // error).
    const mutateFn = vi
      .fn()
      .mockRejectedValue(new Error("Stripe webhook failed"));
    const refetchFn = vi.fn();
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
      mutateFn,
      refetchFn,
    });
    render(<SubscriptionTierSection />);

    const keepButtons = screen.getAllByRole("button", {
      name: /keep max/i,
    });
    fireEvent.click(keepButtons[0]);

    await waitFor(() => {
      expect(screen.getByRole("alert")).toBeDefined();
      expect(screen.getByText(/stripe webhook failed/i)).toBeDefined();
    });
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "Failed to cancel pending change",
        variant: "destructive",
      }),
    );
    expect(refetchFn).toHaveBeenCalled();
  });

  it("disables the tier button that matches the pending tier so users can't overwrite their own scheduled change by mis-click", () => {
    // User is on MAX and has a pending downgrade to PRO. The "Downgrade
    // to Pro" button must be disabled + labelled "Scheduled" so the primary
    // cancel path stays the banner. Other tier buttons (BASIC here) remain
    // clickable — the user can still overwrite their pending change by
    // picking a different target; backend handles that.
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
    });
    render(<SubscriptionTierSection />);

    const scheduledBtn = screen.getByRole("button", { name: /scheduled/i });
    expect(scheduledBtn).toBeDefined();
    expect((scheduledBtn as HTMLButtonElement).disabled).toBe(true);

    // The non-pending tier (BASIC) button is still clickable.
    const basicBtn = screen.getByRole("button", {
      name: /downgrade to basic/i,
    });
    expect((basicBtn as HTMLButtonElement).disabled).toBe(false);
  });

  it("shows replace-pending dialog when clicking a non-pending tier while a pending change exists, and fires the mutation after confirm", async () => {
    // User is on MAX with a pending downgrade to PRO. Clicking BASIC (a
    // tier that is neither current nor the pending target) must NOT silently
    // overwrite the pending schedule — it must open a confirmation dialog.
    // Only after the user explicitly confirms should changeTier (→ its own
    // downgrade confirm for paid→BASIC) fire.
    const mutateFn = vi
      .fn()
      .mockResolvedValue({ status: 200, data: { url: "" } });
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
      mutateFn,
    });
    render(<SubscriptionTierSection />);

    // Clicking BASIC while PRO is pending surfaces the replace-pending dialog
    // before anything mutates.
    fireEvent.click(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    );
    expect(screen.getByRole("dialog")).toBeDefined();
    expect(screen.getByText(/replace pending change/i)).toBeDefined();
    expect(mutateFn).not.toHaveBeenCalled();

    // Confirm the replace: the replace-pending dialog closes and the
    // downgrade-to-BASIC dialog takes over (because BASIC is a downgrade).
    fireEvent.click(
      screen.getByRole("button", { name: /replace pending change/i }),
    );

    // Now the "Confirm Downgrade" dialog should be open — confirm it to fire
    // the mutation.
    fireEvent.click(screen.getByRole("button", { name: /confirm downgrade/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledWith(
        expect.objectContaining({
          data: expect.objectContaining({ tier: "BASIC" }),
        }),
      );
    });
  });

  it("dismisses replace-pending dialog on Cancel without mutating", () => {
    const mutateFn = vi
      .fn()
      .mockResolvedValue({ status: 200, data: { url: "" } });
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "PRO",
        pendingTierEffectiveAt: new Date("2026-11-15T00:00:00Z"),
      }),
      mutateFn,
    });
    render(<SubscriptionTierSection />);

    fireEvent.click(
      screen.getByRole("button", { name: /downgrade to basic/i }),
    );
    expect(screen.getByRole("dialog")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));
    expect(screen.queryByRole("dialog")).toBeNull();
    expect(mutateFn).not.toHaveBeenCalled();
  });

  it("renders Cancel subscription button for paid users with no pending change", () => {
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);
    expect(
      screen.getByRole("button", { name: /cancel subscription/i }),
    ).toBeDefined();
  });

  it("hides Cancel subscription button for NO_TIER users (already cancelled)", () => {
    setupMocks({ subscription: makeSubscription({ tier: "NO_TIER" }) });
    render(<SubscriptionTierSection />);
    expect(
      screen.queryByRole("button", { name: /cancel subscription/i }),
    ).toBeNull();
  });

  it("hides Cancel subscription button when payment flag is disabled", () => {
    mockPaymentEnabled = false;
    setupMocks({ subscription: makeSubscription({ tier: "PRO" }) });
    render(<SubscriptionTierSection />);
    expect(
      screen.queryByRole("button", { name: /cancel subscription/i }),
    ).toBeNull();
  });

  it("hides Cancel subscription button when a pending change is already scheduled", () => {
    // Avoid double-cancelling: PendingChangeBanner exposes the cancel-pending
    // path; an extra Cancel button here would be redundant and confusing.
    setupMocks({
      subscription: makeSubscription({
        tier: "PRO",
        pendingTier: "NO_TIER",
        pendingTierEffectiveAt: new Date("2026-05-15T12:00:00Z"),
      }),
    });
    render(<SubscriptionTierSection />);
    expect(
      screen.queryByRole("button", { name: /cancel subscription/i }),
    ).toBeNull();
  });

  it("opens the cancel-confirm dialog with NO_TIER copy when Cancel subscription is clicked", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "PRO",
        // No current_period_end so the date suffix is absent — keeps the
        // matcher simple and exercises the optional-period branch.
      }),
    });
    render(<SubscriptionTierSection />);
    fireEvent.click(
      screen.getByRole("button", { name: /cancel subscription/i }),
    );
    expect(screen.getByRole("dialog")).toBeDefined();
    expect(
      screen.getByText(/cancelling your subscription schedules it to end/i),
    ).toBeDefined();
  });

  it("renders BASIC cancellation copy in banner when pending_tier is BASIC", () => {
    setupMocks({
      subscription: makeSubscription({
        tier: "MAX",
        pendingTier: "BASIC",
        // Noon UTC so the local-formatted date lands on the same day
        // regardless of the runner's timezone (midnight UTC drifts to
        // the prior day in any timezone west of UTC).
        pendingTierEffectiveAt: new Date("2026-05-15T12:00:00Z"),
      }),
    });
    render(<SubscriptionTierSection />);
    // Cancellation copy — distinct from the generic downgrade phrasing.
    expect(
      screen.getByText(/scheduled to cancel your subscription on/i),
    ).toBeDefined();
    expect(screen.getByText(/May 15, 2026/)).toBeDefined();
    // Must NOT render the "downgrade to" phrasing on BASIC cancellation.
    expect(screen.queryByText(/scheduled to downgrade to/i)).toBeNull();
  });
});
