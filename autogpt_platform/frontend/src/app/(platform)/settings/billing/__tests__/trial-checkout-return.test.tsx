import { act, render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import {
  cancelPending,
  mockBilling,
  trialSubscription,
} from "./trial-plan-fixtures";

const mockSearchParams = vi.hoisted(() => ({
  current: new URLSearchParams(),
}));
vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
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
}));

const FINISHING = "Finishing your new plan…";

function advance(ms: number) {
  return act(() => vi.advanceTimersByTimeAsync(ms));
}

beforeEach(() => {
  setTrialUser();
  vi.useFakeTimers({ shouldAdvanceTime: true });
  mockSearchParams.current = new URLSearchParams({
    subscription: "success",
    session_id: "cs_test_max",
    plan: "MAX",
    cycle: "monthly",
  });
});
afterEach(() => {
  vi.useRealTimers();
  setTrialUser(null);
  mockSearchParams.current = new URLSearchParams();
});

describe("returning from a plan Checkout while the trial is cancel-pending", () => {
  it("offers no second purchase until the new plan replaces the trial", async () => {
    const { state, hits } = mockBilling();
    render(<SettingsBillingPage />);
    expect(await screen.findByText(FINISHING)).toBeDefined();
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();

    state.trial = trialResponse({
      ...cancelPending,
      active: false,
      status: "canceled",
    });
    state.subscription = { ...trialSubscription, tier: "MAX" };
    await advance(2_000);

    expect(await screen.findByText("Active")).toBeDefined();
    expect(screen.getByText("Max")).toBeDefined();
    expect(screen.queryByText(FINISHING)).toBeNull();
    expect(screen.queryByText("Cancellation pending")).toBeNull();
    expect(screen.queryByText("Your trial has ended")).toBeNull();
    const settled = { ...hits };
    await advance(10_000);
    expect(hits.trial).toBe(settled.trial);
    expect(hits.subscription).toBe(settled.subscription);
  });

  it("offers the plan choices again when the plan takes too long", async () => {
    const { hits } = mockBilling();
    render(<SettingsBillingPage />);
    await screen.findByText(FINISHING);
    await advance(10_000);
    expect(hits.subscription).toBeGreaterThan(3);
    expect(hits.trial).toBeGreaterThan(3);

    await advance(25_000);
    expect(
      await screen.findByRole("region", { name: "Plan choices" }),
    ).toBeDefined();
    expect(screen.queryByText(FINISHING)).toBeNull();
    const settled = { ...hits };
    await advance(10_000);
    expect(hits.subscription).toBe(settled.subscription);
    expect(hits.trial).toBe(settled.trial);
  });
});

describe("returning from a plan Checkout without a trial", () => {
  it("does not keep checking the plan", async () => {
    const { state, hits } = mockBilling(
      trialResponse({ active: false, status: "canceled" }),
    );
    state.subscription = { ...trialSubscription, tier: "PRO" };
    render(<SettingsBillingPage />);
    expect(await screen.findByText("Active")).toBeDefined();
    const settled = { ...hits };
    await advance(10_000);
    expect(hits.subscription).toBe(settled.subscription);
    expect(hits.trial).toBe(settled.trial);
    expect(screen.queryByText(FINISHING)).toBeNull();
  });
});
