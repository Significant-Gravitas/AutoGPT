import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { setTrialUser } from "@/tests/integrations/trial-fixtures";
import { AutopilotUsageCard } from "../components/SubscriptionTab/AutopilotUsageCard/AutopilotUsageCard";

beforeEach(() => setTrialUser());
afterEach(() => setTrialUser(null));

// Local times far ahead, so the card dates them the same in any time zone.
const DAILY_RESET = new Date(2030, 10, 5, 20, 30, 56);
const WEEKLY_RESET = new Date(2030, 10, 12, 8, 30, 56);
const EXPIRES = new Date(2030, 11, 4, 20, 30, 56);

function usage(over: Record<string, unknown> = {}) {
  server.use(
    http.get("*/api/credits/subscription", () =>
      HttpResponse.json({ tier: "MAX", monthly_cost: 32000 }),
    ),
    http.get("*/api/credits/trial", () => HttpResponse.json(null)),
    http.get("*/api/chat/usage", () =>
      HttpResponse.json({
        tier: "MAX",
        daily: { percent_used: 40, resets_at: DAILY_RESET.toISOString() },
        weekly: { percent_used: 25, resets_at: WEEKLY_RESET.toISOString() },
        state: "ok",
        ...over,
      }),
    ),
  );
}

it("shows each window's exact reset in the reader's own time, never early", async () => {
  usage();
  render(<AutopilotUsageCard />);

  // 8:30:56 is shown as 8:31, with its date.
  expect(await screen.findByText("Resets Tue, Nov 12, 8:31 AM")).toBeDefined();
  expect(screen.getByText("Resets Tue, Nov 5, 8:31 PM")).toBeDefined();
});

it("shows nothing new for a plan without a reserve, extra week or offer", async () => {
  usage();
  render(<AutopilotUsageCard />);

  await screen.findByText("Resets Tue, Nov 12, 8:31 AM");
  expect(screen.queryByText("Daily reserve")).toBeNull();
  expect(screen.queryByText("Extra week")).toBeNull();
  expect(
    screen.queryByRole("button", { name: /Add an extra week/ }),
  ).toBeNull();
});

it("shows the daily reserve and when it refills", async () => {
  usage({
    reserve: { percent_used: 75, resets_at: DAILY_RESET.toISOString() },
  });
  render(<AutopilotUsageCard />);

  expect(await screen.findByText("Daily reserve")).toBeDefined();
  expect(screen.getByText("75% used today")).toBeDefined();
  expect(screen.getByText(/Refills Tue, Nov 5, 8:31 PM\./)).toBeDefined();
});

it("shows a paid extra week, how much is used and when it expires", async () => {
  usage({
    extra: { percent_used: 62, count: 1, expires_at: EXPIRES.toISOString() },
  });
  render(<AutopilotUsageCard />);

  expect(await screen.findByText("Extra week")).toBeDefined();
  expect(screen.getByText("62% used")).toBeDefined();
  expect(
    screen.getByText(
      "Used once your weekly limit runs out. Expires Wed, Dec 4, 8:31 PM.",
    ),
  ).toBeDefined();
});

it("counts stacked extra weeks", async () => {
  usage({
    extra: { percent_used: 10, count: 3, expires_at: EXPIRES.toISOString() },
  });
  render(<AutopilotUsageCard />);

  expect(await screen.findByText("3 extra weeks")).toBeDefined();
});

it("offers an extra week at the plan's price", async () => {
  usage({ extra_week_offer: { price_credits: 8000 } });
  render(<AutopilotUsageCard />);

  expect(
    await screen.findByRole("button", { name: "Add an extra week · $80" }),
  ).toBeDefined();
  expect(screen.getByText("Out of usage? Add another week now.")).toBeDefined();
});
