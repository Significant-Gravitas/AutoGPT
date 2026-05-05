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
    expect(screen.getByText("Business")).toBeDefined();
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
  it("displays monthly prices by default", () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    // PRO monthly = 5000 cents = $50.00
    expect(screen.getByText("$50.00")).toBeDefined();
    expect(screen.getByText("$320.00")).toBeDefined();
  });

  it("toggling Yearly switches displayed prices to tier_costs_yearly", async () => {
    setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("radio", { name: /yearly/i }));

    // PRO yearly = 51000 cents = $510.00; MAX yearly = 326400 cents = $3,264.00
    await waitFor(() => {
      expect(screen.getByText("$510.00")).toBeDefined();
      expect(screen.getByText("$3,264.00")).toBeDefined();
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

  it("clicking Upgrade to Pro with monthly toggle fires {tier:PRO, billing_cycle:monthly}", async () => {
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
    expect(args.data.billing_cycle).toBe("monthly");
  });

  it("clicking Upgrade to Pro with yearly toggle fires {tier:PRO, billing_cycle:yearly}", async () => {
    stubLocation();
    const { mutateFn } = setupMocks({
      subscription: {
        tier: "NO_TIER",
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
      },
    });

    render(<PaywallModal />);

    fireEvent.click(screen.getByRole("radio", { name: /yearly/i }));
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
