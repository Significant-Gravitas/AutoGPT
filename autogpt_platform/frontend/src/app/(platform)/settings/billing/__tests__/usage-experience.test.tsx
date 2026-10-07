import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  fireEvent,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { YourPlanCard } from "../components/SubscriptionTab/YourPlanCard/YourPlanCard";
import { AutopilotUsageCard } from "../components/SubscriptionTab/AutopilotUsageCard/AutopilotUsageCard";

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

function mockUsage(tier: string, trial = {}, daily = 100, weekly = 100) {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({ tier, monthly_cost: 0 }),
    ),
    http.get("*/api/credits/trial", () =>
      HttpResponse.json(trialResponse(trial)),
    ),
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({
        tier,
        daily: { percent_used: daily, resets_at: "2030-09-13T00:00:00Z" },
        weekly: { percent_used: weekly, resets_at: "2030-09-16T00:00:00Z" },
      }),
    ),
  );
}

it("shows one lifetime trial meter and never describes trial end as a reset", async () => {
  mockUsage("TRIAL", { usage_policy: "lifetime", allowance_used_percent: 100 });
  render(<AutopilotUsageCard />);
  expect(await screen.findByText("Trial usage")).toBeDefined();
  expect(screen.getAllByRole("progressbar")).toHaveLength(1);
  expect(
    screen.getByText(/This allowance lasts for your whole trial/),
  ).toBeDefined();
  expect(screen.queryByText(/Resets/)).toBeNull();
});

it("uses the later blocking window when both paid limits are exhausted", async () => {
  mockUsage("PRO", { active: false, converted: true });
  render(<AutopilotUsageCard />);
  const primary = await screen.findByTestId("featured-usage");
  expect(primary.textContent).toContain("This Week");
  expect(screen.getAllByRole("progressbar")).toHaveLength(2);
});

it("does not display retained trial consumption after conversion", async () => {
  mockUsage(
    "PRO",
    { active: false, converted: true, allowance_used_percent: 100 },
    12,
    35,
  );
  render(<AutopilotUsageCard />);
  expect(await screen.findByText("Today")).toBeDefined();
  expect(screen.queryByText("Trial usage")).toBeNull();
  expect(screen.queryByText("100% used")).toBeNull();
});

it("offers a real retry rather than showing zero after a usage failure", async () => {
  mockUsage("PRO", { active: false, converted: true }, 12, 35);
  let attempts = 0;
  server.use(
    http.get("*/api/chat/usage", () => {
      attempts += 1;
      return attempts === 1
        ? HttpResponse.json({}, { status: 500 })
        : HttpResponse.json({
            tier: "PRO",
            daily: { percent_used: 12, resets_at: null },
            weekly: { percent_used: 35, resets_at: null },
          });
    }),
  );
  render(<AutopilotUsageCard />);
  fireEvent.click(await screen.findByRole("button", { name: /try again/i }));
  await waitFor(() => expect(screen.getByText("Today")).toBeDefined());
  expect(attempts).toBeGreaterThan(1);
});

it("keeps plan recovery visible when an inactive account has no usage permission", async () => {
  mockUsage("NO_TIER", { active: false, converted: true });
  server.use(
    http.get("*/api/chat/usage", () => HttpResponse.json({}, { status: 403 })),
  );
  render(<AutopilotUsageCard />);
  expect(await screen.findByText("Ready for a fresh start?")).toBeDefined();
  expect(screen.queryByRole("progressbar")).toBeNull();
  expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
});

it("explains an unlimited paid plan without inventing zero usage", async () => {
  mockUsage("ENTERPRISE", { active: false, converted: true });
  server.use(
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({ tier: "ENTERPRISE", daily: null, weekly: null }),
    ),
  );
  render(<AutopilotUsageCard />);
  expect(
    await screen.findByText("Your plan has no daily or weekly usage limits."),
  ).toBeDefined();
  expect(screen.queryByRole("progressbar")).toBeNull();
  expect(screen.queryByText("0% used")).toBeNull();
});

it("features the exhausted total allowance for a legacy trial without suggesting a rolling reset restores access", async () => {
  mockUsage(
    "TRIAL",
    { usage_policy: "rolling", allowance_used_percent: 100 },
    25,
    60,
  );
  render(<AutopilotUsageCard />);
  expect(await screen.findByText("Trial usage")).toBeDefined();
  expect(screen.getAllByRole("progressbar")).toHaveLength(1);
  expect(screen.getByRole("progressbar").getAttribute("aria-valuenow")).toBe(
    "100",
  );
  expect(screen.queryByText(/Resets/)).toBeNull();
  expect(screen.queryByText("Today")).toBeNull();
});

it("keeps both recurring meters for a legacy trial whose total allowance remains available", async () => {
  mockUsage(
    "TRIAL",
    { usage_policy: "rolling", allowance_used_percent: 42 },
    100,
    60,
  );
  render(<AutopilotUsageCard />);
  expect(await screen.findByText("Today")).toBeDefined();
  expect(screen.getByText("This Week")).toBeDefined();
  expect(screen.getAllByRole("progressbar")).toHaveLength(2);
  expect(screen.queryByText("Trial usage")).toBeNull();
});

it("refreshes the rendered usage after a Max upgrade without resetting consumption", async () => {
  mockUsage("PRO", { active: false, converted: true });
  let upgraded = false;
  let usageRequests = 0;
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({
        tier: upgraded ? "MAX" : "PRO",
        monthly_cost: upgraded ? 32000 : 5000,
        tier_costs: { PRO: 5000, MAX: 32000 },
        billing_cycle: "monthly",
        has_active_stripe_subscription: true,
      }),
    ),
    http.get("*/api/credits/manage", () =>
      HttpResponse.json({ url: "https://billing.stripe.com/p/test" }),
    ),
    http.get("*/api/chat/usage", () => {
      usageRequests += 1;
      return HttpResponse.json({
        tier: upgraded ? "MAX" : "PRO",
        daily: {
          percent_used: upgraded ? 12 : 100,
          resets_at: "2030-09-13T00:00:00Z",
        },
        weekly: {
          percent_used: upgraded ? 10 : 85,
          resets_at: "2030-09-16T00:00:00Z",
        },
      });
    }),
    http.post("*/api/credits/subscription", () => {
      upgraded = true;
      return HttpResponse.json({ url: "" });
    }),
  );
  render(<YourPlanCard showUsage />);
  await waitFor(() =>
    expect(
      screen
        .getByRole("progressbar", { name: "Today usage" })
        .getAttribute("aria-valuenow"),
    ).toBe("100"),
  );
  fireEvent.click(screen.getByRole("button", { name: "Upgrade to Max" }));
  fireEvent.click(
    within(await screen.findByRole("dialog")).getByRole("button", {
      name: "Upgrade to Max",
    }),
  );
  await waitFor(() =>
    expect(
      screen
        .getByRole("progressbar", { name: "Today usage" })
        .getAttribute("aria-valuenow"),
    ).toBe("12"),
  );
  expect(usageRequests).toBeGreaterThanOrEqual(2);
  expect(screen.queryByText("0% used")).toBeNull();
});
