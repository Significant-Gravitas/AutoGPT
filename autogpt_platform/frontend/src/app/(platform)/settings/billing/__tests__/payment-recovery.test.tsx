import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import SettingsBillingPage from "../page";

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

const PAYMENT_URL = "https://invoice.stripe.com/i/acct_1/test_inv_1";

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

function mockPastDueTrial(status: string, paymentDue: boolean) {
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(
      trialResponse({
        active: false,
        status,
        payment_due: paymentDue
          ? { amount_due: 5413, currency: "usd", payment_url: PAYMENT_URL }
          : null,
      }),
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
}

it.each(["past_due", "unpaid"])(
  "gives a %s trial one message and one way to pay the open invoice",
  async (status) => {
    mockPastDueTrial(status, true);
    render(<SettingsBillingPage />);
    const pay = await screen.findByRole("link", { name: "Pay $54.13 now" });
    expect(pay.getAttribute("href")).toBe(PAYMENT_URL);
    expect(pay.getAttribute("target")).toBe("_blank");
    expect(screen.getByText("Your payment didn’t go through")).toBeDefined();
    expect(
      screen.getByText(
        "Pro was due to start on Sep 17, but the $54.13 charge didn’t go through. Pay now and Pro starts right away. Your chats, workflows and results are saved.",
      ),
    ).toBeDefined();
    expect(screen.queryByText("Your trial has ended")).toBeNull();
    expect(
      screen.queryByText("Pick a plan to continue using AutoGPT."),
    ).toBeNull();
    expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
    expect(screen.queryByRole("region", { name: "Usage" })).toBeNull();
    expect(screen.queryByText("Update payment method")).toBeNull();
    expect(screen.getByText("Manage payment method")).toBeDefined();
  },
);

it("points to the payment method row when the open invoice can't be read", async () => {
  mockPastDueTrial("past_due", false);
  render(<SettingsBillingPage />);
  expect(
    await screen.findByText(
      "Update your payment method below to continue. Your chats, workflows and results are saved.",
    ),
  ).toBeDefined();
  expect(screen.queryByRole("link", { name: /^Pay / })).toBeNull();
  expect(screen.queryByRole("button", { name: /^Start/ })).toBeNull();
  expect(screen.getByText("Manage payment method")).toBeDefined();
});
