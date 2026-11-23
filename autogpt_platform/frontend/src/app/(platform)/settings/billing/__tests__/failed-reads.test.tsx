import { delay, http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { resetClientRequestFailureLog } from "@/app/api/mutators/request-failure-log";
import { server } from "@/mocks/mock-server";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import SettingsBillingPage from "../page";

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

const mockSearchParams = { current: new URLSearchParams() };
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => mockSearchParams.current,
    useRouter: () => ({
      push: vi.fn(),
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/settings/billing",
    useParams: () => ({}),
  };
});

// Each answer takes a round trip of fake time, as in a browser, so a page
// that reads again whenever a read fails does about one read per round trip.
const ROUND_TRIP_MS = 100;
const WINDOW_MS = 5_000;

let reads: Record<string, number>;

function countRead({ request }: { request: Request }) {
  const path = new URL(request.url).pathname.replace("/api/proxy", "");
  reads[path] = (reads[path] ?? 0) + 1;
}

beforeEach(() => {
  vi.useFakeTimers();
  setTrialUser();
  reads = {};
  server.events.on("request:start", countRead);
});

afterEach(async () => {
  server.events.removeListener("request:start", countRead);
  // Unmount first, then land the answers still on their way, so a failed
  // test leaves no handler waiting on a fake timer that teardown removes.
  cleanup();
  await vi.advanceTimersByTimeAsync(ROUND_TRIP_MS);
  vi.useRealTimers();
  setTrialUser(null);
  mockSearchParams.current = new URLSearchParams();
  resetClientRequestFailureLog();
});

function answer(path: string, status: number, body: object) {
  return http.get(`*/api/${path}`, async () => {
    await delay(ROUND_TRIP_MS);
    return HttpResponse.json(body, { status });
  });
}

function failing(path: string, status: number) {
  return answer(path, status, { detail: `failed with ${status}` });
}

function plan(tier: string) {
  return answer("credits/subscription", 200, {
    tier,
    monthly_cost: tier === "PRO" ? 5000 : 0,
    has_active_stripe_subscription: tier !== "NO_TIER",
    tier_costs: { PRO: 5000, MAX: 32000 },
    tier_multipliers: { PRO: 1, MAX: 8.5 },
  });
}

async function openBillingFor(milliseconds: number) {
  render(<SettingsBillingPage />);
  await act(() => vi.advanceTimersByTimeAsync(milliseconds));
}

it.each([401, 403, 429, 500, 502, 503, 504])(
  "reads a trial status that fails with %s once more when the plan shows, not in a loop",
  async (status) => {
    server.use(
      failing("credits/trial", status),
      plan("PRO"),
      answer("credits/invoices", 200, []),
    );
    await openBillingFor(WINDOW_MS);

    expect(reads["/api/credits/trial"]).toBe(2);
    expect(reads["/api/credits/subscription"]).toBe(2);
    expect(reads["/api/chat/usage"]).toBe(1);
    expect(
      screen.getByText(
        "We had the following error when retrieving trial information:",
      ),
    ).toBeDefined();
    expect(screen.getByRole("heading", { name: "Pro" })).toBeDefined();
    expect(screen.getByRole("region", { name: "Usage" })).toBeDefined();
  },
);

it("keeps the plan to pick when the trial status of an account with no plan fails", async () => {
  server.use(
    failing("credits/trial", 500),
    plan("NO_TIER"),
    answer("credits/invoices", 200, []),
  );
  await openBillingFor(WINDOW_MS);

  expect(reads["/api/credits/trial"]).toBe(2);
  expect(reads["/api/credits/subscription"]).toBe(2);
  expect(
    screen.getByText("Pick a plan to continue using AutoGPT."),
  ).toBeDefined();
});

it("reads a failing plan once more when a checkout return shows the trial's plan choices, not in a loop", async () => {
  mockSearchParams.current = new URLSearchParams(
    "subscription=success&plan=PRO&cycle=monthly",
  );
  server.use(
    answer("credits/trial", 200, trialResponse({ cancel_at_period_end: true })),
    failing("credits/subscription", 500),
    answer("credits/invoices", 200, []),
  );
  await openBillingFor(WINDOW_MS);

  expect(reads["/api/credits/subscription"]).toBe(2);
  expect(screen.getByRole("button", { name: "Resume trial" })).toBeDefined();
  expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
});
