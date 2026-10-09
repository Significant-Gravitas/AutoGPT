import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import SettingsBillingPage from "../page";

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

it.each(["past_due", "unpaid", "incomplete_expired", "trialing", "canceled"])(
  "keeps plan selection available for an inactive %s trial",
  async (status) => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ active: false, status }),
      ),
      http.get("*/api/credits/subscription", () =>
        HttpResponse.json({
          tier: "NO_TIER",
          monthly_cost: 0,
          has_active_stripe_subscription: false,
        }),
      ),
      http.get("*/api/credits/invoices", () => HttpResponse.json([])),
    );
    render(<SettingsBillingPage />);
    await screen.findByText("Your trial has ended");
    expect(
      await screen.findByText("Pick a plan to continue using AutoGPT."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Get Pro" })).toBeDefined();
  },
);

it("shows the trial as the current plan during an active trial", async () => {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier: "TRIAL",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
      }),
    ),
    http.get("*/api/credits/invoices", () => HttpResponse.json([])),
  );
  render(<SettingsBillingPage />);
  await screen.findByRole("button", { name: "Cancel trial" });
  expect(screen.getByText("Your plan")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Get Pro" })).toBeNull();
});

const trialSubscription = {
  tier: "TRIAL",
  monthly_cost: 0,
  proration_credit_cents: 0,
  has_active_stripe_subscription: true,
  tier_costs: { PRO: 5000, MAX: 32000 },
  tier_multipliers: { PRO: 1, MAX: 8.5 },
};

function mockBilling(trial: ReturnType<typeof trialResponse>, tier: string) {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(trial),
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({ ...trialSubscription, tier }),
    ),
    http.get("*/api/credits/invoices", () => HttpResponse.json([])),
  );
}

it.each([true, false])(
  "offers plan choices while a trial cancellation is pending (cancel_keeps_access %s)",
  async (keepsAccess) => {
    mockBilling(
      trialResponse({
        cancel_at_period_end: true,
        cancel_keeps_access: keepsAccess,
      }),
      "TRIAL",
    );
    render(<SettingsBillingPage />);
    expect(
      await screen.findByRole("button", { name: "Subscribe to Pro" }),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Upgrade to Max" }),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Get Pro" })).toBeNull();
  },
);

it("offers no plan choices during a trial that will convert", async () => {
  mockBilling(
    trialResponse({ cancel_at_period_end: false, cancel_keeps_access: true }),
    "TRIAL",
  );
  render(<SettingsBillingPage />);
  await screen.findByRole("button", { name: "Cancel trial" });
  expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Subscribe to Pro" })).toBeNull();
});

it("offers no plan choices once a cancel-pending trial has converted", async () => {
  mockBilling(
    trialResponse({
      cancel_at_period_end: true,
      cancel_keeps_access: true,
      converted: true,
    }),
    "PRO",
  );
  render(<SettingsBillingPage />);
  await screen.findByRole("button", { name: "Upgrade to Max" });
  expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Subscribe to Pro" })).toBeNull();
});

it("offers no plan choices once a cancel-pending trial has ended", async () => {
  mockBilling(
    trialResponse({
      active: false,
      status: "canceled",
      cancel_at_period_end: true,
      cancel_keeps_access: true,
    }),
    "NO_TIER",
  );
  render(<SettingsBillingPage />);
  await screen.findByRole("button", { name: "Get Pro" });
  expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
  expect(screen.queryByRole("button", { name: "Subscribe to Pro" })).toBeNull();
});
