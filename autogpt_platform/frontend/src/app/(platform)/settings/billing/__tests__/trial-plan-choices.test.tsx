import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import {
  cancelPending,
  mockBilling,
  trialSubscription,
  yearlyOffer,
} from "./trial-plan-fixtures";

const analytics = vi.hoisted(() => ({
  trackPaywallViewed: vi.fn(),
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

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  analytics.trackPaywallViewed.mockReset();
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
