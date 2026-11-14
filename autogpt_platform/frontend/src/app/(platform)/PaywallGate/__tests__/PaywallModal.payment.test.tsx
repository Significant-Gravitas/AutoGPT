import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const mockUseGetSubscriptionStatus = vi.fn();
const mockUseUpdateSubscriptionTier = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: unknown) =>
    mockUseGetSubscriptionStatus(opts),
  useUpdateSubscriptionTier: () => mockUseUpdateSubscriptionTier(),
  getGetSubscriptionStatusQueryKey: () => ["/api/credits/subscription"],
}));

vi.mock("@posthog/react", () => ({
  useFeatureFlagVariantKey: () => undefined,
  usePostHog: () => ({ capture: vi.fn() }),
}));

function MockDialog({
  title,
  children,
}: {
  title: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <div role="dialog">
      <div>{title}</div>
      {children}
    </div>
  );
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

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    validateSession: vi.fn().mockResolvedValue(true),
    isLoggedIn: true,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/copilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { PaywallModal } from "../PaywallModal";

const PAYMENT_URL = "https://invoice.stripe.com/i/acct_1/test_inv_1";

beforeEach(() => {
  setTrialUser();
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: { tier: "NO_TIER", tier_costs: { PRO: 5000, MAX: 32000 } },
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
  });
  mockUseUpdateSubscriptionTier.mockReturnValue({
    mutateAsync: vi.fn(),
    isPending: false,
  });
});

afterEach(() => {
  setTrialUser(null);
  vi.restoreAllMocks();
});

test("a failed first trial charge pays the open invoice instead of choosing a plan", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({
        active: false,
        status: "past_due",
        payment_due: {
          amount_due: 5000,
          currency: "usd",
          payment_url: PAYMENT_URL,
        },
      }),
    ),
  );

  render(<PaywallModal />);

  const pay = await screen.findByRole("link", { name: "Pay $50 now" });
  expect(pay.getAttribute("href")).toBe(PAYMENT_URL);
  expect(screen.getByText("Your payment didn’t go through")).toBeDefined();
  expect(screen.queryByText("Choose a plan to keep working")).toBeNull();
  expect(screen.queryByRole("region", { name: "Pro plan" })).toBeNull();
  expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
  expect(screen.getByRole("button", { name: "Back to billing" })).toBeDefined();
});

test("sends a failed charge without a readable invoice to billing", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, status: "past_due", payment_due: null }),
    ),
  );

  render(<PaywallModal />);

  expect(
    await screen.findByText(
      "Update your payment method in billing to continue. Your chats, workflows and results are saved.",
    ),
  ).toBeDefined();
  expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
  expect(screen.getByRole("button", { name: "Back to billing" })).toBeDefined();
});

test("keeps plan choices for a trial that simply ended", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, status: "expired" }),
    ),
  );

  render(<PaywallModal />);

  expect(await screen.findByRole("region", { name: "Pro plan" })).toBeDefined();
  expect(screen.queryByText("Your payment didn’t go through")).toBeNull();
});

const RENEWAL_BILL = {
  amount_due: 5000,
  currency: "usd",
  payment_url: PAYMENT_URL,
  plan: "PRO",
  due_since: Date.UTC(2026, 2, 3, 12) / 1000,
};

function subscriptionWithBill(payment_due: typeof RENEWAL_BILL | null) {
  return {
    data: {
      tier: "NO_TIER",
      tier_costs: { PRO: 5000, MAX: 32000 },
      payment_due,
    },
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
  };
}

test("a failed renewal pays its open bill instead of offering Get Pro", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, converted: true, status: "past_due" }),
    ),
  );
  mockUseGetSubscriptionStatus.mockReturnValue(
    subscriptionWithBill(RENEWAL_BILL),
  );

  render(<PaywallModal />);

  const pay = await screen.findByRole("link", { name: "Pay $50 now" });
  expect(pay.getAttribute("href")).toBe(PAYMENT_URL);
  expect(
    screen.getByText(/Pro was due to renew on Mar 3, but the \$50 charge/),
  ).toBeDefined();
  expect(
    screen.getByText(/a full billing period from the day you pay/),
  ).toBeDefined();
  expect(screen.queryByRole("region", { name: "Pro plan" })).toBeNull();
  expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
  expect(screen.getByRole("button", { name: "Back to billing" })).toBeDefined();
});

test("keeps checking the subscription while a renewal bill is open", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, converted: true, status: "past_due" }),
    ),
  );
  mockUseGetSubscriptionStatus.mockReturnValue(
    subscriptionWithBill(RENEWAL_BILL),
  );

  render(<PaywallModal />);
  await screen.findByRole("link", { name: "Pay $50 now" });

  const recoveryQuery = mockUseGetSubscriptionStatus.mock.calls
    .map(([options]) => options as { query?: Record<string, unknown> })
    .find((options) => options.query?.enabled === true);
  const interval = recoveryQuery?.query?.refetchInterval as (query: {
    state: { data?: unknown };
  }) => number | false;
  expect(
    interval({
      state: { data: { status: 200, data: { payment_due: RENEWAL_BILL } } },
    }),
  ).toBe(15_000);
  expect(
    interval({ state: { data: { status: 200, data: { payment_due: null } } } }),
  ).toBe(false);
});

test("offers plans again once nothing is left to pay", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ active: false, status: "expired" }),
    ),
  );
  mockUseGetSubscriptionStatus.mockReturnValue(subscriptionWithBill(null));

  render(<PaywallModal />);

  expect(await screen.findByRole("region", { name: "Pro plan" })).toBeDefined();
  expect(screen.queryByText("Your payment didn’t go through")).toBeNull();
});
