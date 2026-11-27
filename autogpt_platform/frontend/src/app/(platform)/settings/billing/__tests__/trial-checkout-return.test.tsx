import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import { getGetTrialsGetTrialStatusQueryKey } from "@/app/api/__generated__/endpoints/trials/trials";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { onlineManager, QueryClient } from "@tanstack/react-query";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import {
  cancelPending,
  mockBilling,
  trialSubscription,
} from "./trial-plan-fixtures";

// These cover the hosted billing UI, which self-hosted builds hide.
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

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

// What tests hold back: answers to plan reads, and the network when a test
// takes it down. Teardown unmounts the page, then releases whatever a failed
// test left held, so none outlives its test.
const heldAnswers: Array<() => Promise<void>> = [];

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
afterEach(async () => {
  cleanup();
  await Promise.all(heldAnswers.splice(0).map((answer) => answer()));
  vi.useRealTimers();
  setTrialUser(null);
  mockSearchParams.current = new URLSearchParams();
});

// Holds every plan read until the test answers them, then answers with the
// plan `state` holds by then.
function holdPlanReads(state: ReturnType<typeof mockBilling>["state"]) {
  const release = Promise.withResolvers<void>();
  const answered = Promise.withResolvers<void>();
  let reads = 0;
  heldAnswers.push(async () => {
    release.resolve();
    if (reads > 0) await answered.promise;
  });
  server.use(
    http.get("*/api/credits/subscription", async () => {
      reads += 1;
      await release.promise;
      answered.resolve();
      return HttpResponse.json(state.subscription);
    }),
  );
  return () => release.resolve();
}

// Takes the network down until the test brings it back. Teardown drops the
// reads still waiting for it before bringing it back, so none goes out after
// the test.
function takeOffline(queryClient: QueryClient) {
  onlineManager.setOnline(false);
  heldAnswers.push(async () => {
    queryClient.clear();
    onlineManager.setOnline(true);
  });
  return () => onlineManager.setOnline(true);
}

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

  it("offers no second purchase while the first plan read is on its way", async () => {
    const { state } = mockBilling();
    const answerPlan = holdPlanReads(state);
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    render(<SettingsBillingPage />, { queryClient });

    expect(await screen.findByText(FINISHING)).toBeDefined();
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();

    answerPlan();
    await advance(500);
    expect(
      queryClient.getQueryState(getGetSubscriptionStatusQueryKey())?.status,
    ).toBe("success");
    expect(screen.getByText(FINISHING)).toBeDefined();
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
  });

  it("offers no second purchase while the first plan read waits for the network", async () => {
    mockBilling();
    const queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    // The trial has answered; only the plan read is left waiting.
    queryClient.setQueryData(
      [
        ...getGetTrialsGetTrialStatusQueryKey(),
        useAuthStore.getState().user?.id,
      ],
      {
        status: 200,
        data: trialResponse(cancelPending),
        headers: new Headers(),
      },
    );
    const goOnline = takeOffline(queryClient);
    render(<SettingsBillingPage />, { queryClient });

    expect(await screen.findByText(FINISHING)).toBeDefined();
    expect(
      queryClient.getQueryState(getGetSubscriptionStatusQueryKey())
        ?.fetchStatus,
    ).toBe("paused");
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
    expect(screen.queryByRole("button", { name: "Upgrade to Max" })).toBeNull();

    goOnline();
    await advance(500);
    expect(
      queryClient.getQueryState(getGetSubscriptionStatusQueryKey())?.status,
    ).toBe("success");
    expect(screen.getByText(FINISHING)).toBeDefined();
    expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
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
