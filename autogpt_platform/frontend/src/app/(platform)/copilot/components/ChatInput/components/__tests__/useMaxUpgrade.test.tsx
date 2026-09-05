import { getGetV2ListChatConnectionsQueryKey } from "@/app/api/__generated__/endpoints/chat/chat";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import type { ReactNode } from "react";
import { beforeEach, describe, expect, it } from "vitest";

import { useMaxUpgrade } from "../ConnectionPicker/useMaxUpgrade";

let current: SubscriptionStatusResponse;
let posts: unknown[];
let reads: number;

function setup(enabled = true) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>{children}</QueryClientProvider>
    );
  }
  return {
    client,
    ...renderHook(() => useMaxUpgrade(enabled), { wrapper: Wrapper }),
  };
}

beforeEach(() => {
  current = {
    tier: "PRO",
    monthly_cost: 3100,
    tier_costs: { PRO: 3100, MAX: 17900 },
    tier_costs_yearly: { PRO: 31000, MAX: 179000 },
    billing_cycle: "monthly",
    tier_multipliers: { PRO: 1, MAX: 4 },
    proration_credit_cents: 1000,
    has_active_stripe_subscription: true,
  };
  posts = [];
  reads = 0;
  server.use(
    http.get("*/api/credits/subscription", () => {
      reads++;
      return HttpResponse.json(current);
    }),
    http.post("*/api/credits/subscription", async ({ request }) => {
      posts.push(await request.json());
      current = { ...current, tier: "MAX" };
      return HttpResponse.json(current);
    }),
  );
});

describe("useMaxUpgrade", () => {
  it("does not request subscription data before the locked picker opens", () => {
    const { result } = setup(false);
    expect(reads).toBe(0);
    expect(result.current.canUpgrade).toBe(false);
  });

  it.each(["monthly", "yearly"] as const)(
    "confirms Max on the current %s billing cycle and refreshes connections",
    async (cycle) => {
      current.billing_cycle = cycle;
      const { result, client } = setup();
      const connectionsKey = getGetV2ListChatConnectionsQueryKey();
      client.setQueryData(connectionsKey, { offers: [] });
      await waitFor(() => expect(result.current.canUpgrade).toBe(true));
      expect(posts).toHaveLength(0);
      let upgraded = false;
      await act(async () => {
        upgraded = await result.current.upgrade();
      });
      expect(upgraded).toBe(true);
      expect(posts).toEqual([{ tier: "MAX", billing_cycle: cycle }]);
      expect(client.getQueryState(connectionsKey)?.isInvalidated).toBe(true);
      expect(result.current.subscription?.tier).toBe("MAX");
    },
  );

  it("does not charge twice when confirmation is invoked twice", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.canUpgrade).toBe(true));
    await act(async () => {
      await Promise.all([result.current.upgrade(), result.current.upgrade()]);
    });
    expect(posts).toHaveLength(1);
  });

  it.each([
    { has_active_stripe_subscription: false },
    { pending_tier: "NO_TIER" as const },
    { billing_cycle: "yearly" as const, tier_costs_yearly: {} },
    { tier: "MAX" as const },
  ])(
    "requires billing fallback for unavailable account state %j",
    async (state) => {
      Object.assign(current, state);
      const { result } = setup();
      await waitFor(() => expect(result.current.isLoading).toBe(false));
      expect(result.current.canUpgrade).toBe(false);
      await act(async () => {
        expect(await result.current.upgrade()).toBe(false);
      });
      expect(posts).toHaveLength(0);
    },
  );

  it("refreshes changed prices before charging and requires a new review", async () => {
    const { result } = setup();
    await waitFor(() => expect(result.current.canUpgrade).toBe(true));
    current = { ...current, tier_costs: { PRO: 3100, MAX: 22900 } };
    await act(async () => {
      expect(await result.current.upgrade()).toBe(false);
    });
    expect(posts).toHaveLength(0);
    expect(result.current.pricing?.maxCents).toBe(22900);
  });

  it("recovers from a load error without offering a blind charge", async () => {
    server.use(
      http.get(
        "*/api/credits/subscription",
        () => new HttpResponse(null, { status: 503 }),
      ),
    );
    const { result } = setup();
    await waitFor(() => expect(result.current.isError).toBe(true));
    expect(result.current.canUpgrade).toBe(false);
    server.use(
      http.get("*/api/credits/subscription", () => HttpResponse.json(current)),
    );
    act(() => result.current.retry());
    await waitFor(() => expect(result.current.canUpgrade).toBe(true));
  });

  it("keeps the offer open for a declined payment without claiming success", async () => {
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json(
          { detail: "Your card was declined." },
          { status: 402 },
        ),
      ),
    );
    const { result } = setup();
    await waitFor(() => expect(result.current.canUpgrade).toBe(true));
    await act(async () => {
      expect(await result.current.upgrade()).toBe(false);
    });
    expect(result.current.isPending).toBe(false);
    expect(result.current.subscription?.tier).toBe("PRO");
    expect(result.current.error).toMatch(/card was declined/i);
    act(() => result.current.resetError());
    expect(result.current.error).toBeNull();
  });

  it("does not navigate away or claim success for an unexpected checkout response", async () => {
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json({
          ...current,
          url: "https://checkout.stripe.com/example",
        }),
      ),
    );
    const { result } = setup();
    const originalURL = window.location.href;
    await waitFor(() => expect(result.current.canUpgrade).toBe(true));
    await act(async () => {
      expect(await result.current.upgrade()).toBe(false);
    });
    expect(window.location.href).toBe(originalURL);
  });
});
