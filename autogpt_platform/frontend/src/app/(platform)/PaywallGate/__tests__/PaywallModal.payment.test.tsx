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
