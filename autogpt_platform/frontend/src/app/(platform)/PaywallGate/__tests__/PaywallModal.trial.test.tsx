import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";

const mockUseGetSubscriptionStatus = vi.fn();
const mockUseUpdateSubscriptionTier = vi.fn();
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: unknown) =>
    mockUseGetSubscriptionStatus(opts),
  useUpdateSubscriptionTier: () => mockUseUpdateSubscriptionTier(),
}));

const posthog = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("@posthog/react", () => ({
  useFeatureFlagVariantKey: () => undefined,
  usePostHog: () => posthog,
}));

// Radix portals don't render under happy-dom; keep the dialog tree inline.
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
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/build",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
}));

import { PaywallModal } from "../PaywallModal";

const originalLocation = window.location;
let checkoutLocation = {
  origin: "https://app.test",
  href: "https://app.test/build",
  assign: vi.fn(),
};

function setSubscription(tierCosts: Record<string, number>) {
  mockUseGetSubscriptionStatus.mockReturnValue({
    data: { tier: "NO_TIER", tier_costs: tierCosts },
    isLoading: false,
    isFetching: false,
    refetch: vi.fn(),
  });
  mockUseUpdateSubscriptionTier.mockReturnValue({
    mutateAsync: vi.fn().mockResolvedValue({ status: 200, data: { url: "" } }),
    isPending: false,
  });
}

function eligible(offer = trialOffer) {
  return trialResponse({ eligible: true, active: false, status: null, offer });
}

beforeEach(() => {
  setTrialUser();
  posthog.capture.mockClear();
  setSubscription({ PRO: 5000, MAX: 32000 });
  checkoutLocation = {
    origin: "https://app.test",
    href: "https://app.test/build",
    assign: vi.fn(),
  };
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: checkoutLocation,
  });
});

afterEach(() => {
  setTrialUser(null);
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
  vi.restoreAllMocks();
});

test.each([
  ["PRO", "Pro"],
  ["MAX", "Max"],
] as const)(
  "starts the %s trial from the paywall and returns to billing",
  async (tier, name) => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        eligible({ ...trialOffer, tier }),
      ),
    );
    const requestBody = vi.fn();
    server.use(
      getPostTrialsStartTrialCheckoutMockHandler200(async ({ request }) => {
        requestBody(await request.json());
        return { url: "https://checkout.stripe.com/trial" };
      }),
    );

    render(<PaywallModal />);

    const card = within(
      await screen.findByRole("region", { name: `${name} plan` }),
    );
    fireEvent.click(
      await card.findByRole("button", { name: "Start 7-day trial" }),
    );

    await waitFor(() =>
      expect(checkoutLocation.assign).toHaveBeenCalledWith(
        "https://checkout.stripe.com/trial",
      ),
    );
    // The paywall gates every non-exempt route, so Stripe returns the user to
    // /settings/billing, which confirms the trial. By then the enrollment has
    // set a real tier and the gate has lifted.
    expect(requestBody).toHaveBeenCalledExactlyOnceWith({
      offer_token: trialOffer.token,
      return_to: "billing",
    });
    expect(posthog.capture).toHaveBeenCalledWith(
      "subscription_trial_checkout_started",
      { trial_offer_version: trialOffer.version, surface: "billing" },
    );
  },
);

test("reports the offer impression against the billing surface", async () => {
  server.use(getGetTrialsGetTrialStatusMockHandler200(eligible()));

  render(<PaywallModal />);

  await screen.findByRole("button", { name: "Start 7-day trial" });
  expect(posthog.capture).toHaveBeenCalledWith(
    "subscription_trial_offer_viewed",
    expect.objectContaining({ surface: "billing" }),
  );
});

test("keeps paid checkout reporting against the upgrade_modal surface", async () => {
  server.use(getGetTrialsGetTrialStatusMockHandler200(eligible()));

  render(<PaywallModal />);

  const max = within(await screen.findByRole("region", { name: "Max plan" }));
  const paid = max.getByRole("button", { name: "Upgrade to Max" });
  expect(paid.getAttribute("data-fast-goal")).toBe("plan_cta_click");
  expect(paid.getAttribute("data-fast-goal-surface")).toBe("upgrade_modal");

  const toggle = screen.getByRole("button", { name: /yearly/i });
  expect(toggle.getAttribute("data-fast-goal-surface")).toBe("upgrade_modal");
});

test("leaves the paid CTAs alone when the user is not eligible", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({ eligible: false, converted: true }),
    ),
  );

  render(<PaywallModal />);

  const pro = within(await screen.findByRole("region", { name: "Pro plan" }));
  expect(pro.getByRole("button", { name: "Upgrade to Pro" })).toBeDefined();
  expect(screen.queryByRole("button", { name: /Start .*trial/ })).toBeNull();
});

test("leaves the paid CTAs alone when the trial status fails to load", async () => {
  server.use(
    http.get("*/api/credits/trial", () =>
      HttpResponse.json({ detail: "boom" }, { status: 500 }),
    ),
  );

  render(<PaywallModal />);

  const pro = within(await screen.findByRole("region", { name: "Pro plan" }));
  expect(pro.getByRole("button", { name: "Upgrade to Pro" })).toBeDefined();
  expect(screen.queryByRole("button", { name: /Start .*trial/ })).toBeNull();
});

test("hides an offer for a tier the backend isn't pricing on this paywall", async () => {
  // LD can hide MAX from /credits/subscription while a MAX trial offer is
  // still live — the trial CTA would have no card to sit on.
  setSubscription({ PRO: 5000 });
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      eligible({ ...trialOffer, tier: "MAX" }),
    ),
  );

  render(<PaywallModal />);

  const pro = within(await screen.findByRole("region", { name: "Pro plan" }));
  expect(pro.getByRole("button", { name: "Upgrade to Pro" })).toBeDefined();
  expect(screen.queryByRole("button", { name: /Start .*trial/ })).toBeNull();
});

test("offers to switch billing cycle when the trial is priced on the other one", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      eligible({ ...trialOffer, billing_cycle: "yearly" }),
    ),
  );

  render(<PaywallModal />);

  const pro = within(await screen.findByRole("region", { name: "Pro plan" }));
  const swap = await pro.findByRole("button", {
    name: "Try Pro for 7 days instead",
  });
  fireEvent.click(swap);

  expect(
    await pro.findByRole("button", { name: "Start 7-day trial" }),
  ).toBeDefined();
});
