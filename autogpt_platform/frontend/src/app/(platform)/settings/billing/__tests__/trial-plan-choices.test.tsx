import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const analytics = vi.hoisted(() => ({
  trackPaywallViewed: vi.fn(),
  trackPlanSelected: vi.fn(),
}));
vi.mock(
  "@/services/analytics/monetization-analytics",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/services/analytics/monetization-analytics")
    >()),
    ...analytics,
  }),
);

const cancelPending = { cancel_at_period_end: true, cancel_keeps_access: true };

const trialSubscription: SubscriptionStatusResponse = {
  tier: "TRIAL",
  monthly_cost: 0,
  proration_credit_cents: 0,
  has_active_stripe_subscription: true,
  tier_costs: { PRO: 5000, MAX: 32000 },
  tier_costs_yearly: { PRO: 51000, MAX: 326400 },
  tier_multipliers: { PRO: 1, MAX: 8.5 },
};

const yearlyOffer: TrialOfferResponse = {
  ...trialOffer,
  billing_cycle: "yearly",
  unit_amount: 51000,
};

function returnURLs(plan: string, cycle: string) {
  const page = `${window.location.origin}${window.location.pathname}`;
  return {
    success_url: `${page}?subscription=success&session_id={CHECKOUT_SESSION_ID}&plan=${plan}&cycle=${cycle}`,
    cancel_url: `${page}?subscription=cancelled`,
  };
}

function mockBilling(trial = trialResponse(cancelPending)) {
  const state = { trial, subscription: trialSubscription };
  const hits = { trial: 0, subscription: 0 };
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(() => {
      hits.trial += 1;
      return state.trial;
    }),
    http.get("*/api/credits/subscription", () => {
      hits.subscription += 1;
      return HttpResponse.json(state.subscription);
    }),
    http.get("*/api/credits/invoices", () => HttpResponse.json([])),
  );
  return { state, hits };
}

function mockPlanRequest(respond: () => Response) {
  const body = vi.fn();
  server.use(
    http.post("*/api/credits/subscription", async ({ request }) => {
      body(await request.json());
      return respond();
    }),
  );
  return body;
}

async function openConfirmation() {
  fireEvent.click(
    await screen.findByRole("button", { name: "Subscribe to Pro" }),
  );
  return screen.findByRole("dialog", { name: "Start Pro today?" });
}

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  toast.mockReset();
  analytics.trackPaywallViewed.mockReset();
  analytics.trackPlanSelected.mockReset();
  removeGtagShim();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("plan choices while a trial cancellation is pending", () => {
  it("prices the trial's plan and the next tier up from the status data", async () => {
    mockBilling();
    render(<SettingsBillingPage />);
    const choices = await screen.findByRole("region", {
      name: "Plan choices",
    });
    const [own, upgrade] = within(choices).getAllByRole("listitem");
    expect(within(own).getByText("Pro")).toBeDefined();
    expect(within(own).getByText("$20.00")).toBeDefined();
    expect(within(own).getByText("/ month")).toBeDefined();
    expect(
      within(own).getByText(
        "Starts today. Your trial ends and the plan takes over.",
      ),
    ).toBeDefined();
    expect(
      within(own).getByRole("button", { name: "Subscribe to Pro" }),
    ).toBeDefined();
    expect(within(upgrade).getByText("Max")).toBeDefined();
    expect(within(upgrade).getByText("$320.00")).toBeDefined();
    expect(within(upgrade).getByText("/ month")).toBeDefined();
    expect(
      within(upgrade).getByText(
        "8.5x the usage of Pro for people who run a lot.",
      ),
    ).toBeDefined();
    expect(
      within(upgrade).getByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
    expect(analytics.trackPaywallViewed).toHaveBeenCalledWith("billing");
  });

  it("prices both plans on the trial's yearly cycle", async () => {
    mockBilling(trialResponse({ ...cancelPending, offer: yearlyOffer }));
    render(<SettingsBillingPage />);
    const choices = await screen.findByRole("region", {
      name: "Plan choices",
    });
    const [own, upgrade] = within(choices).getAllByRole("listitem");
    expect(within(own).getByText("$510.00")).toBeDefined();
    expect(within(own).getByText("/ year")).toBeDefined();
    expect(within(upgrade).getByText("$3,264.00")).toBeDefined();
    expect(within(upgrade).getByText("/ year")).toBeDefined();
  });

  it("leaves out a next tier with no price on the trial's cycle", async () => {
    const { state } = mockBilling(
      trialResponse({ ...cancelPending, offer: yearlyOffer }),
    );
    state.subscription = { ...trialSubscription, tier_costs_yearly: {} };
    render(<SettingsBillingPage />);
    expect(
      await screen.findByRole("button", { name: "Subscribe to Pro" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();
  });

  it("falls back to plain copy when usage multipliers are missing", async () => {
    const { state } = mockBilling();
    state.subscription = { ...trialSubscription, tier_multipliers: {} };
    render(<SettingsBillingPage />);
    expect(
      await screen.findByText("More usage than Pro for people who run a lot."),
    ).toBeDefined();
  });

  it("shows nothing until plan prices load", async () => {
    const prices = deferredTrialResponse<void>();
    const { hits } = mockBilling();
    server.use(
      http.get("*/api/credits/subscription", async () => {
        hits.subscription += 1;
        await prices.promise;
        return HttpResponse.json(trialSubscription);
      }),
    );
    render(<SettingsBillingPage />);
    await waitFor(() => expect(hits.subscription).toBeGreaterThan(0));
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    prices.resolve();
    expect(
      await screen.findByRole("region", { name: "Plan choices" }),
    ).toBeDefined();
  });

  it("shows nothing when plan prices fail to load", async () => {
    const { hits } = mockBilling();
    server.use(
      http.get("*/api/credits/subscription", () => {
        hits.subscription += 1;
        return HttpResponse.json({ detail: "unavailable" }, { status: 500 });
      }),
    );
    render(<SettingsBillingPage />);
    await waitFor(() => expect(hits.subscription).toBeGreaterThan(0));
    await new Promise((settle) => setTimeout(settle, 50));
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    expect(
      screen.queryByRole("button", { name: "Subscribe to Pro" }),
    ).toBeNull();
  });
});

describe("subscribing to the trial's plan", () => {
  it("charges today only after confirming, then refreshes trial and plan", async () => {
    const { state, hits } = mockBilling();
    const body = mockPlanRequest(() => {
      state.trial = trialResponse({ ...cancelPending, converted: true });
      state.subscription = { ...trialSubscription, tier: "PRO" };
      return HttpResponse.json({ ...state.subscription, url: "" });
    });
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    expect(
      within(dialog).getByText(
        "Your trial ends now and your saved card is charged $20.00 / month, plus applicable tax.",
      ),
    ).toBeDefined();
    expect(body).not.toHaveBeenCalled();
    expect(analytics.trackPlanSelected).toHaveBeenCalledWith({
      subscription_tier: "PRO",
      billing_cycle: "monthly",
      surface: "billing",
    });
    const before = { ...hits };

    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );

    await waitFor(() => expect(body).toHaveBeenCalledOnce());
    expect(body).toHaveBeenCalledWith({
      tier: "PRO",
      billing_cycle: "monthly",
      surface: "billing",
      ...returnURLs("PRO", "monthly"),
    });
    await waitFor(() => expect(hits.trial).toBeGreaterThan(before.trial));
    await waitFor(() =>
      expect(hits.subscription).toBeGreaterThan(before.subscription),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "You're on Pro" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull(),
    );
  });

  it("sends the trial's own cycle for a yearly trial", async () => {
    mockBilling(trialResponse({ ...cancelPending, offer: yearlyOffer }));
    const body = mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: "" }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    expect(
      within(dialog).getByText(
        "Your trial ends now and your saved card is charged $510.00 / year, plus applicable tax.",
      ),
    ).toBeDefined();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );
    await waitFor(() =>
      expect(body).toHaveBeenCalledWith({
        tier: "PRO",
        billing_cycle: "yearly",
        surface: "billing",
        ...returnURLs("PRO", "yearly"),
      }),
    );
  });

  it("keeps the trial without a request when the person backs out", async () => {
    mockBilling();
    const body = mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: "" }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Keep my trial" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(body).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Subscribe to Pro" }),
    ).toBeDefined();
  });

  it("shows why the charge failed and keeps the choices usable", async () => {
    mockBilling();
    const body = mockPlanRequest(() =>
      HttpResponse.json({ detail: "Your card was declined." }, { status: 402 }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Your card was declined.");
    expect(body).toHaveBeenCalledOnce();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(toast).not.toHaveBeenCalled();
    expect(
      screen
        .getByRole("button", { name: "Subscribe to Pro" })
        .hasAttribute("disabled"),
    ).toBe(false);
  });
});

describe("upgrading past the trial's plan", () => {
  it.each([
    ["monthly", trialOffer],
    ["yearly", yearlyOffer],
  ] as const)(
    "opens Stripe Checkout for the next tier on the %s cycle",
    async (cycle, offer) => {
      const assign = vi
        .spyOn(window.location, "assign")
        .mockImplementation(() => {});
      mockBilling(trialResponse({ ...cancelPending, offer }));
      const body = mockPlanRequest(() =>
        HttpResponse.json({
          ...trialSubscription,
          url: "https://checkout.stripe.com/c/pay/cs_max",
        }),
      );
      render(<SettingsBillingPage />);
      fireEvent.click(
        await screen.findByRole("button", { name: "Upgrade to Max" }),
      );

      await waitFor(() =>
        expect(assign).toHaveBeenCalledWith(
          "https://checkout.stripe.com/c/pay/cs_max",
        ),
      );
      expect(body).toHaveBeenCalledWith({
        tier: "MAX",
        billing_cycle: cycle,
        surface: "billing",
        ...returnURLs("MAX", cycle),
      });
      expect(analytics.trackPlanSelected).toHaveBeenCalledWith({
        subscription_tier: "MAX",
        billing_cycle: cycle,
        surface: "billing",
      });
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(toast).not.toHaveBeenCalled();
    },
  );

  it("reports begin_checkout at the next tier's price before leaving", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
    const gtagCalls = installGtagShim();
    const assign = vi
      .spyOn(window.location, "assign")
      .mockImplementation(() => {});
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json({
        ...trialSubscription,
        url: "https://checkout.stripe.com/c/pay/cs_max",
      }),
    );
    render(<SettingsBillingPage />);
    const upgrade = await screen.findByRole("button", {
      name: "Upgrade to Max",
    });
    fireEvent.click(upgrade);

    await waitFor(() =>
      expect(gtagCalls).toContainEqual([
        "event",
        "conversion",
        {
          send_to: "AW-123/BC",
          value: 320,
          currency: "USD",
          event_callback: expect.any(Function),
        },
      ]),
    );
    expect(upgrade.hasAttribute("disabled")).toBe(true);
    await waitFor(() =>
      expect(assign).toHaveBeenCalledWith(
        "https://checkout.stripe.com/c/pay/cs_max",
      ),
    );
  });

  it("shows why Checkout could not start", async () => {
    const assign = vi
      .spyOn(window.location, "assign")
      .mockImplementation(() => {});
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json(
        { detail: "Stripe is unavailable. Please retry." },
        { status: 502 },
      ),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Stripe is unavailable. Please retry.");
    expect(assign).not.toHaveBeenCalled();
    expect(
      screen
        .getByRole("button", { name: "Upgrade to Max" })
        .hasAttribute("disabled"),
    ).toBe(false);
  });

  it("does not send user B to the Checkout user A started", async () => {
    const assign = vi
      .spyOn(window.location, "assign")
      .mockImplementation(() => {});
    setTrialUser("user-a");
    mockBilling();
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        trialResponse({
          ...cancelPending,
          offer: {
            ...trialOffer,
            unit_amount:
              useAuthStore.getState().user?.id === "user-b" ? 3000 : 2000,
          },
        }),
      ),
    );
    const checkout = deferredTrialResponse<void>();
    const sent = vi.fn();
    server.use(
      http.post("*/api/credits/subscription", async () => {
        sent();
        await checkout.promise;
        return HttpResponse.json({
          ...trialSubscription,
          url: "https://checkout.stripe.com/c/pay/cs_user_a",
        });
      }),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    );
    await waitFor(() => expect(sent).toHaveBeenCalledOnce());

    act(() => setTrialUser("user-b"));
    await screen.findByText("$30.00");
    checkout.resolve();
    await act(() => new Promise((settle) => setTimeout(settle, 50)));

    expect(assign).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
