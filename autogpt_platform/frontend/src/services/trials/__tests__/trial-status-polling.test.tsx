import {
  QueryClient,
  QueryClientProvider,
  focusManager,
} from "@tanstack/react-query";
import { act, cleanup, render } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { server } from "@/mocks/mock-server";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { TrialStatusSync } from "../TrialStatusSync";
import { useTrialStatus } from "../useTrialStatus";

vi.mock("@/lib/auth/hooks/useAuthStore", async () => {
  const { create } = await import("zustand");
  return {
    useAuthStore: create<{ user: { id: string } | null }>(() => ({
      user: null,
    })),
  };
});

let trial: TrialStatusResponse;
let client: QueryClient;
const requests = vi.fn();

function Consumer() {
  const status = useTrialStatus();
  return <span>{status.data?.status}</span>;
}

function CheckoutReturnConsumer() {
  useTrialStatus({
    refetchInterval: (status) =>
      status?.active && status.cancel_at_period_end ? 2_000 : false,
  });
  return null;
}

function tree(consumers = 1) {
  return (
    <QueryClientProvider client={client}>
      <TrialStatusSync />
      {Array.from({ length: consumers }, (_, index) => (
        <Consumer key={index} />
      ))}
    </QueryClientProvider>
  );
}

async function advance(milliseconds: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(milliseconds);
  });
}

async function settled() {
  await act(async () => {
    await vi.waitFor(() => expect(client.isFetching()).toBe(0));
    await vi.advanceTimersByTimeAsync(1);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  focusManager.setFocused(true);
  useAuthStore.setState({
    user: { id: "user-A", email: "a@example.com", user_metadata: {} },
  });
  trial = { active: false, eligible: true, status: null };
  requests.mockReset();
  client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  server.use(
    http.get("*/api/proxy/api/credits/trial", () => {
      requests(useAuthStore.getState().user?.id);
      return HttpResponse.json(trial);
    }),
  );
});

afterEach(() => {
  cleanup();
  client.clear();
  focusManager.setFocused(undefined);
  vi.useRealTimers();
});

describe("trial status synchronization", () => {
  it("does not give individual consumers their own polling timers", async () => {
    trial = { active: true, status: "trialing" };
    render(
      <QueryClientProvider client={client}>
        <Consumer />
        <Consumer />
      </QueryClientProvider>,
    );
    await settled();
    await advance(90_000);
    expect(requests).toHaveBeenCalledTimes(1);
  });

  it.each([
    { active: false, eligible: true, status: null },
    { active: false, converted: true, status: "converted" },
    { active: false, status: "canceled" },
    { active: false, eligible: true, status: "checkout_pending" },
  ])("does not poll a stable state %j", async (state) => {
    trial = state;
    render(tree(3));
    await settled();
    expect(requests).toHaveBeenCalledTimes(1);
    await advance(120_000);
    expect(requests).toHaveBeenCalledTimes(1);
  });

  it("has one polling owner even when consumers mount at different times", async () => {
    trial = { active: true, status: "trialing" };
    const view = render(tree());
    await settled();
    await advance(5_000);
    view.rerender(tree(3));
    await advance(25_000);
    await settled();
    expect(requests).toHaveBeenCalledTimes(2);
    await advance(5_000);
    expect(requests).toHaveBeenCalledTimes(2);
    await advance(25_000);
    await settled();
    expect(requests).toHaveBeenCalledTimes(3);
  });

  it("starts polling after a focus refresh activates a trial and stops after conversion", async () => {
    render(tree());
    await settled();
    act(() => focusManager.setFocused(false));
    await advance(11_000);
    trial = { active: true, status: "trialing" };
    act(() => focusManager.setFocused(true));
    await settled();
    expect(requests).toHaveBeenCalledTimes(2);
    trial = { active: false, converted: true, status: "converted" };
    await advance(30_000);
    await settled();
    expect(requests).toHaveBeenCalledTimes(3);
    await advance(90_000);
    expect(requests).toHaveBeenCalledTimes(3);
  });

  it("pauses hidden-tab polling and refreshes once when the user returns", async () => {
    trial = { active: true, status: "trialing" };
    render(tree(3));
    await settled();
    act(() => focusManager.setFocused(false));
    await advance(90_000);
    expect(requests).toHaveBeenCalledTimes(1);
    act(() => focusManager.setFocused(true));
    await settled();
    expect(requests).toHaveBeenCalledTimes(2);
  });

  it("switches cache ownership with the account and stops all requests on sign-out", async () => {
    trial = { active: true, status: "trialing" };
    render(tree(2));
    await settled();
    trial = { active: false, converted: true, status: "converted" };
    act(() =>
      useAuthStore.setState({
        user: { id: "user-B", email: "b@example.com", user_metadata: {} },
      }),
    );
    await settled();
    expect(requests.mock.calls).toEqual([["user-A"], ["user-B"]]);
    await advance(60_000);
    expect(requests).toHaveBeenCalledTimes(2);
    act(() => useAuthStore.setState({ user: null }));
    await advance(60_000);
    act(() => focusManager.setFocused(false));
    act(() => focusManager.setFocused(true));
    await settled();
    expect(requests).toHaveBeenCalledTimes(2);
  });

  it("keeps checkout reconciliation fast, pauses hidden requests, and stops when the plan lands", async () => {
    trial = { active: true, cancel_at_period_end: true, status: "trialing" };
    render(
      <QueryClientProvider client={client}>
        <CheckoutReturnConsumer />
        <Consumer />
      </QueryClientProvider>,
    );
    await settled();
    await advance(2_000);
    await settled();
    expect(requests).toHaveBeenCalledTimes(2);
    act(() => focusManager.setFocused(false));
    await advance(20_000);
    expect(requests).toHaveBeenCalledTimes(2);
    trial = { active: false, converted: true, status: "converted" };
    act(() => focusManager.setFocused(true));
    await settled();
    expect(requests).toHaveBeenCalledTimes(3);
    await advance(60_000);
    expect(requests).toHaveBeenCalledTimes(3);
  });
});
