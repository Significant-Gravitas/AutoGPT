import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { BillingOffer } from "../components/SubscriptionTab/BillingOffer";
import { YourPlanCard } from "../components/SubscriptionTab/YourPlanCard/YourPlanCard";

beforeEach(() => {
  setTrialUser("billing-user");
  server.use(
    http.get("*/api/credits/manage", () =>
      HttpResponse.json({ url: "https://billing.stripe.com/test" }),
    ),
  );
});
afterEach(() => setTrialUser(null));

function mockTrialAt(percent: number) {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({ tier: "TRIAL", monthly_cost: 0 }),
    ),
    http.get("*/api/credits/trial", () =>
      HttpResponse.json(
        trialResponse({
          usage_policy: "lifetime",
          allowance_used_percent: percent,
        }),
      ),
    ),
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({ tier: "TRIAL", daily: null, weekly: null }),
    ),
  );
}

it.each([0, 64, 79])(
  "keeps the price and start date without pushing an early start at %s%%",
  async (percent) => {
    mockTrialAt(percent);
    render(<BillingOffer />);
    expect(
      await screen.findByRole("heading", { name: "Pro starts on Sep 17" }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Your trial keeps going until then. You can also start today.",
      ),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Start Pro now" })).toBeDefined();
    expect(screen.getByText("$20")).toBeDefined();
    expect(screen.queryByText(/keep working today/)).toBeNull();
  },
);

it.each([80, 100])(
  "offers starting Pro today once the trial is running low at %s%%",
  async (percent) => {
    mockTrialAt(percent);
    render(<BillingOffer />);
    expect(
      await screen.findByRole("button", { name: "Start Pro now · $20" }),
    ).toBeDefined();
    expect(
      screen.getByText(
        "Pro starts on its own on Sep 17. Start it now to keep working today.",
      ),
    ).toBeDefined();
  },
);

it("shows one usage error and keeps the normal plan card when usage fails", async () => {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier: "PRO",
        monthly_cost: 5000,
        tier_costs: { PRO: 5000, MAX: 32000 },
        tier_multipliers: { PRO: 1, MAX: 8.5 },
        billing_cycle: "monthly",
        has_active_stripe_subscription: true,
      }),
    ),
    http.get("*/api/credits/trial", () =>
      HttpResponse.json(trialResponse({ active: false, converted: true })),
    ),
    http.get("*/api/chat/usage", () => HttpResponse.json({}, { status: 500 })),
  );
  render(<YourPlanCard showUsage />);
  expect(await screen.findByText("We couldn’t load your usage")).toBeDefined();
  expect(screen.getAllByText("We couldn’t load your usage")).toHaveLength(1);
  expect(screen.getByRole("button", { name: "Try again" })).toBeDefined();
  expect(screen.getByRole("link", { name: "Contact support" })).toBeDefined();
  expect(screen.queryByText("Something went wrong")).toBeNull();
  expect(
    screen.queryByRole("button", { name: /Report Error|Get Help/ }),
  ).toBeNull();
  expect(
    await screen.findByRole("heading", { name: "Upgrade to Max" }),
  ).toBeDefined();
  expect(
    screen.getByText(
      "What you’ve already used counts toward Max’s higher limit.",
    ),
  ).toBeDefined();
});
