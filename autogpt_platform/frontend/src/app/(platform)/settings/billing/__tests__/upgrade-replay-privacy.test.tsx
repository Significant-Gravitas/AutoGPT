import { fireEvent, within } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
  trialOffer,
} from "@/tests/integrations/trial-fixtures";
import { useProActivation } from "@/services/pro-activation/useProActivation";
import { quote } from "@/services/pro-activation/__tests__/fixtures";
import { BillingOffer } from "../components/SubscriptionTab/BillingOffer";
import { YourPlanCard } from "../components/SubscriptionTab/YourPlanCard/YourPlanCard";
import { maxUpgradePreview } from "./max-upgrade.fixture";

beforeEach(() => {
  setTrialUser("billing-user");
  sessionStorage.clear();
  server.use(
    http.get("*/api/credits/manage", () =>
      HttpResponse.json({ url: "https://billing.stripe.com/test" }),
    ),
  );
});
afterEach(() => setTrialUser(null));

it("keeps every reviewed Max amount out of replay, including the consent button and renewal copy", async () => {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier: "PRO",
        monthly_cost: 5000,
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_costs_yearly: { PRO: 51000, MAX: 326400 },
        billing_cycle: "monthly",
        has_active_stripe_subscription: true,
      }),
    ),
    http.post("*/api/credits/subscription/preview", () =>
      HttpResponse.json(
        maxUpgradePreview({ adjustment_amount: 100, amount_due: 27378 }),
      ),
    ),
  );
  render(<YourPlanCard />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Upgrade to Max" }),
  );
  const dialog = within(
    await screen.findByRole("dialog", { name: /Upgrade to Max/ }),
  );
  await dialog.findByRole("button", { name: "Pay $273.78 and upgrade" });
  const amounts = dialog.getAllByText(/\$/);
  expect(amounts).toHaveLength(8);
  for (const amount of amounts) {
    expect(amount.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
  }
});

function StartPro() {
  const activation = useProActivation();
  return <button onClick={() => activation.start()}>Start Pro</button>;
}

it("keeps Pro's reviewed charge, discount, tax and full renewal terms masked in the real dialog", async () => {
  server.use(
    http.get(
      "*/api/credits/pro-activation/current",
      () => new HttpResponse(null, { status: 404 }),
    ),
    http.post("*/api/credits/pro-activation/preview", () =>
      HttpResponse.json({
        ...quote,
        terms: {
          ...quote.terms,
          renewal_terms:
            "Renews at £85.00 per year, plus tax. Cancel before renewal.",
          renewal_discounts: [
            { amount_off: 1200, currency: "gbp", duration: "once" },
          ],
        },
      }),
    ),
  );
  render(<StartPro />);
  fireEvent.click(screen.getByRole("button", { name: "Start Pro" }));
  const dialog = within(
    await screen.findByRole("dialog", { name: /Start Pro today/ }),
  );
  await dialog.findByRole("button", { name: "Pay £61.23 and start Pro" });
  const privateValues = dialog.getAllByText(/£|VAT:/);
  expect(privateValues.length).toBeGreaterThanOrEqual(9);
  for (const value of privateValues) {
    expect(value.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
  }
});

it("masks the accepted trial offer in billing", async () => {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({ tier: "TRIAL", monthly_cost: 0 }),
    ),
    http.get("*/api/credits/trial", () =>
      HttpResponse.json(
        trialResponse({
          usage_policy: "lifetime",
          allowance_used_percent: 100,
          offer: { ...trialOffer, unit_amount: 5013 },
        }),
      ),
    ),
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({ tier: "TRIAL", daily: null, weekly: null }),
    ),
  );
  render(<BillingOffer />);
  await screen.findByRole("button", { name: "Start Pro now · $50.13" });
  const amounts = screen.getAllByText(/\$50\.13/);
  expect(amounts).toHaveLength(2);
  for (const amount of amounts) {
    expect(amount.closest(".sentry-unmask, [data-sentry-unmask]")).toBeNull();
  }
});
