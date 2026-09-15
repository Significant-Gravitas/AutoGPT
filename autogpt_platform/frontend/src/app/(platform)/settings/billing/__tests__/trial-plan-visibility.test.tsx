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

it("keeps the regular plan card hidden during an active trial", async () => {
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
  expect(screen.queryByText("Your plan")).toBeNull();
  expect(screen.queryByRole("button", { name: "Get Pro" })).toBeNull();
});
