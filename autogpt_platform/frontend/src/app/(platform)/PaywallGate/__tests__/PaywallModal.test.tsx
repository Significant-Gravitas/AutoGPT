import { afterEach, describe, expect, it, vi } from "vitest";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  cleanup,
} from "@/tests/integrations/test-utils";

const mockUseGetSubscriptionStatus = vi.fn();
const mockUseUpdateSubscriptionTier = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: unknown) =>
    mockUseGetSubscriptionStatus(opts),
  useUpdateSubscriptionTier: () => mockUseUpdateSubscriptionTier(),
}));

const mockToast = vi.fn();
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast: (args: unknown) => mockToast(args),
  useToast: () => ({ toast: mockToast }),
  useToastOnFail: () => mockToast,
}));

// Strip Radix portals — happy-dom doesn't render them. The mock keeps the
// Dialog tree visible while ignoring controlled/forceOpen props.
function MockDialog({ children }: { children: React.ReactNode }) {
  return <div role="dialog">{children}</div>;
}
function MockDialogContent({ children }: { children: React.ReactNode }) {
  return <div>{children}</div>;
}
function MockDialogFooter({ children }: { children: React.ReactNode }) {
  return <div>{children}</div>;
}
MockDialog.Content = MockDialogContent;
MockDialog.Footer = MockDialogFooter;
vi.mock("@/components/molecules/Dialog/Dialog", () => ({
  Dialog: MockDialog,
}));

import { PaywallModal } from "../PaywallModal";

interface SubscriptionShape {
  tier: string;
  tier_costs: Record<string, number>;
  tier_costs_yearly?: Record<string, number>;
  monthly_cost?: number;
  proration_credit_cents?: number;
  has_active_stripe_subscription?: boolean;
}

function setupMocks({
  subscription,
  isLoading = false,
  mutateFn = vi.fn().mockResolvedValue({ status: 200, data: { url: "" } }),
  isPending = false,
}: {
  subscription: SubscriptionShape | null;
  isLoading?: boolean;
  mutateFn?: ReturnType<typeof vi.fn>;
  isPending?: boolean;
}) {
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: subscription,
    isLoading,
  });
  mockUseUpdateSubscriptionTier.mockReturnValue({
    mutateAsync: mutateFn,
    isPending,
  });
  return { mutateFn };
}

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe("PaywallModal — dynamic plan rendering", () => {
  it("renders one card per tier in tier_costs (PRO + MAX → 2 cards)", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Max")).toBeDefined();
    expect(screen.queryByText("Business")).toBeNull();
  });

  it("renders three cards when tier_costs includes BUSINESS", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000, BUSINESS: 100000 },
        tier_costs_yearly: {
          PRO: 51000,
          MAX: 326400,
          BUSINESS: 1020000,
        },
      },
    });

    render(<PaywallModal />);

    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.getByText("Max")).toBeDefined();
    // BUSINESS is rendered as "Team" — contact-sales tier label.
    expect(screen.getByText("Team")).toBeDefined();
  });

  it("hides cards for tiers absent from tier_costs (LD-hidden tier)", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000 },
      },
    });

    render(<PaywallModal />);

    expect(screen.getByText("Pro")).toBeDefined();
    expect(screen.queryByText("Max")).toBeNull();
    expect(screen.queryByText("Business")).toBeNull();
  });
});

describe("PaywallModal — Monthly/Yearly cycle toggle", () => {
  it("defaults to yearly billing with the monthly-equivalent price and the annual total", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    // PRO yearly = 51000 cents → $42.50/mo primary, $510.00 charged today.
    // MAX yearly = 326400 cents → $272.00/mo primary, $3,264.00 charged today.
    expect(screen.getByText("$42.50")).toBeDefined();
    expect(screen.getByText("$272.00")).toBeDefined();
    expect(screen.getByText("Charged today: $510.00")).toBeDefined();
    expect(screen.getByText("Charged today: $3,264.00")).toBeDefined();
  });

  it("toggling Monthly switches displayed prices to the full monthly amounts", async () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("radio", { name: /monthly/i }));

    // PRO monthly = 5000 cents = $50.00 (primary), $50.00 charged today
    await waitFor(() => {
      expect(screen.getByText("$50.00")).toBeDefined();
      expect(screen.getByText("$320.00")).toBeDefined();
      expect(screen.getByText("Charged today: $50.00")).toBeDefined();
      expect(screen.getByText("Charged today: $320.00")).toBeDefined();
    });
  });
});

describe("PaywallModal — upgrade mutation", () => {
  const originalLocation = window.location;

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
  });

  function stubLocation() {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { origin: "https://app.test", href: "" },
    });
  }

  it("clicking Upgrade to Pro after switching to monthly fires {tier:PRO, billing_cycle:monthly}", async () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("radio", { name: /monthly/i }));
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledTimes(1);
    });
    const [args] = mutateFn.mock.calls[0];
    expect(args.data.tier).toBe("PRO");
    expect(args.data.billing_cycle).toBe("monthly");
  });

  it("clicking Upgrade to Pro with the default yearly toggle fires {tier:PRO, billing_cycle:yearly}", async () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledTimes(1);
    });
    const [args] = mutateFn.mock.calls[0];
    expect(args.data.tier).toBe("PRO");
    expect(args.data.billing_cycle).toBe("yearly");
  });

  it("redirects to Stripe when the response includes a checkout URL", async () => {
    stubLocation();
    const mutateFn = vi.fn().mockResolvedValue({
      status: 200,
      data: { url: "https://checkout.stripe.com/c/test" },
    });
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
      },
      mutateFn,
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    await waitFor(() => {
      expect(window.location.href).toBe("https://checkout.stripe.com/c/test");
    });
  });

  it("Team (BUSINESS) opens the contact-sales intake form instead of firing updateTier", async () => {
    stubLocation();
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        // Including BUSINESS in tier_costs so the card renders; click should
        // still divert to the intake form rather than POSTing to /credits/subscription.
        tier_costs: { PRO: 5000, MAX: 32000, BUSINESS: 50000 },
      },
    });

    render(<PaywallModal />);

    // BUSINESS card renders the "Talk to sales" CTA per PLAN_METADATA.
    fireEvent.click(screen.getByRole("button", { name: /talk to sales/i }));

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining("tally.so"),
      "_blank",
      "noopener,noreferrer",
    );
    expect(mutateFn).not.toHaveBeenCalled();
  });

  it("422 from updateTier surfaces a toast and does not redirect", async () => {
    stubLocation();
    const mutateFn = vi
      .fn()
      .mockRejectedValue(new Error("Unprocessable Entity"));
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
      },
      mutateFn,
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    await waitFor(() => {
      expect(mockToast).toHaveBeenCalledTimes(1);
    });
    expect(window.location.href).toBe("");
  });
});

describe("PaywallModal — admin-overridden NO_TIER with active Stripe sub", () => {
  const originalLocation = window.location;

  afterEach(() => {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: originalLocation,
    });
  });

  function stubLocation() {
    Object.defineProperty(window, "location", {
      configurable: true,
      writable: true,
      value: { origin: "https://app.test", href: "" },
    });
  }

  it("staging gate: clicking Upgrade with active Stripe sub opens confirm dialog and skips mutation", () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        has_active_stripe_subscription: true,
      } as SubscriptionShape,
    });

    render(<PaywallModal />);
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    // Confirmation dialog text surfaces; mutation has NOT fired yet.
    expect(
      screen.getByText(
        /current Stripe subscription will be modified — you may be charged or refunded/i,
      ),
    ).toBeDefined();
    expect(mutateFn).not.toHaveBeenCalled();
  });

  it("confirming the dialog fires the upgrade mutation", async () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        has_active_stripe_subscription: true,
      } as SubscriptionShape,
    });

    render(<PaywallModal />);
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));
    fireEvent.click(screen.getByRole("button", { name: /switch to pro/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledTimes(1);
    });
    const [args] = mutateFn.mock.calls[0];
    expect(args.data.tier).toBe("PRO");
  });

  it("cancelling the dialog leaves the user on the paywall and does not fire", () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        has_active_stripe_subscription: true,
      } as SubscriptionShape,
    });

    render(<PaywallModal />);
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));
    fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

    expect(mutateFn).not.toHaveBeenCalled();
  });

  it("no active Stripe sub: clicking Upgrade fires the mutation directly (Checkout flow)", async () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        has_active_stripe_subscription: false,
      } as SubscriptionShape,
    });

    render(<PaywallModal />);
    fireEvent.click(screen.getByRole("button", { name: /upgrade to pro/i }));

    await waitFor(() => {
      expect(mutateFn).toHaveBeenCalledTimes(1);
    });
  });
});

describe("PaywallModal — empty / loading states", () => {
  it("renders the temporarily-unavailable fallback when tier_costs is empty", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: {},
      },
    });

    render(<PaywallModal />);

    expect(
      screen.getByText(/Subscriptions are temporarily unavailable/i),
    ).toBeDefined();
    // No upgrade buttons, no cycle toggle, no plan cards.
    expect(screen.queryByRole("radio", { name: /monthly/i })).toBeNull();
    expect(screen.queryByRole("button", { name: /upgrade to/i })).toBeNull();
  });

  it("renders skeletons while subscription status is loading", () => {
    setupMocks({ subscription: null, isLoading: true });

    render(<PaywallModal />);

    // Skeleton has no a11y role; assert no cards rendered.
    expect(screen.queryByRole("button", { name: /upgrade to/i })).toBeNull();
    expect(
      screen.queryByText(/Subscriptions are temporarily unavailable/i),
    ).toBeNull();
  });
});
