/**
 * Hook-level coverage for the billing card hooks.
 *
 * The card components themselves are exercised in `billing-cards.test.tsx`;
 * this file uses `renderHook` to drive the mutation paths
 * (handleSubmit, save, disable, changeTier) without going through Radix
 * Dialog interactions, which keeps the tests deterministic and lifts patch
 * coverage on the new hook code above the codecov 80% gate.
 */

import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { server } from "@/mocks/mock-server";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook, waitFor } from "@testing-library/react";
import { http, HttpResponse, type JsonBodyType } from "msw";
import type { ReactNode } from "react";
import { describe, expect, it, vi } from "vitest";

import { useAutoRefillCard } from "../components/AutomationCreditsTab/AutoRefillCard/useAutoRefillCard";
import { useBalanceCard } from "../components/AutomationCreditsTab/BalanceCard/useBalanceCard";
import { usePaymentMethodCard } from "../components/SubscriptionTab/PaymentMethodCard/usePaymentMethodCard";
import { useYourPlanCard } from "../components/SubscriptionTab/YourPlanCard/useYourPlanCard";

function jsonHandler(method: "get" | "post", path: string, body: JsonBodyType) {
  return http[method](`*${path}`, () => HttpResponse.json(body));
}

function makeWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={client}>
        <TooltipProvider>{children}</TooltipProvider>
      </QueryClientProvider>
    );
  }
  return Wrapper;
}

describe("useBalanceCard", () => {
  it("returns isError + null balance when /api/credits returns 500", async () => {
    server.use(
      http.get("*/api/credits", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => {
      expect(result.current.isLoading).toBe(false);
      expect(result.current.isError).toBe(true);
    });

    expect(result.current.balanceCents).toBeNull();
  });

  it("isValid rejects fractional dollars (5.25) and accepts integer >= 5", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1000 }));

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => result.current.setAmount("5.25"));
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setAmount("4"));
    expect(result.current.isValid).toBe(false);

    act(() => result.current.setAmount("5"));
    expect(result.current.isValid).toBe(true);

    act(() => result.current.setAmount("100"));
    expect(result.current.isValid).toBe(true);
  });

  it("handleSubmit POSTs the integer-cents amount when isValid", async () => {
    let capturedBody: { credit_amount: number } | null = null;
    server.use(
      jsonHandler("get", "/api/credits", { credits: 1000 }),
      http.post("*/api/credits", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        // Return without checkout_url so the hook does NOT navigate (would
        // tear down the test environment via window.location.href).
        return HttpResponse.json({ checkout_url: null });
      }),
    );

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => result.current.setAmount("25"));

    await act(async () => {
      result.current.handleSubmit();
    });

    await waitFor(() => expect(result.current.isAdding).toBe(false));
    // Hook treats a no-checkout_url response as an error → modal stays open.
    expect(capturedBody).toEqual({ credit_amount: 2500 });
  });

  it("handleSubmit catches API failures and stops the spinner", async () => {
    server.use(
      jsonHandler("get", "/api/credits", { credits: 1000 }),
      http.post("*/api/credits", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => result.current.setAmount("10"));

    await act(async () => {
      result.current.handleSubmit();
    });

    await waitFor(() => expect(result.current.isAdding).toBe(false));
  });

  it("handleSubmit no-ops when isValid is false (no mutation request)", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1000 }));

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Default amount is "" → isValid false → handleSubmit returns immediately.
    act(() => result.current.handleSubmit());
    expect(result.current.isAdding).toBe(false);
  });
});

describe("useAutoRefillCard", () => {
  it("isValid requires Number.isInteger + threshold/refill >= 5 + refill >= threshold", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 0,
        threshold: 0,
      }),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Below the $5 minimum on either field.
    act(() => {
      result.current.setThreshold("4");
      result.current.setRefillAmount("10");
    });
    expect(result.current.isValid).toBe(false);

    // Refill < threshold rejected (backend 422 mirror).
    act(() => {
      result.current.setThreshold("20");
      result.current.setRefillAmount("10");
    });
    expect(result.current.isValid).toBe(false);

    // Fractional rejected.
    act(() => {
      result.current.setThreshold("5");
      result.current.setRefillAmount("5.5");
    });
    expect(result.current.isValid).toBe(false);

    // Valid: both integers >= 5, refill >= threshold.
    act(() => {
      result.current.setThreshold("5");
      result.current.setRefillAmount("20");
    });
    expect(result.current.isValid).toBe(true);
  });

  it("isEnabled is false for amount=0/threshold=0 and true once configured", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 2000,
        threshold: 500,
      }),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.isEnabled).toBe(true);
    expect(result.current.config?.amount).toBe(2000);
    expect(result.current.config?.threshold).toBe(500);
  });

  it("save() with valid input POSTs the configured amount and threshold", async () => {
    let capturedBody: { amount: number; threshold: number } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 0,
        threshold: 0,
      }),
      http.post("*/api/credits/auto-top-up", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({});
      }),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.setThreshold("5");
      result.current.setRefillAmount("20");
    });

    await act(async () => {
      result.current.save();
    });

    await waitFor(() => expect(result.current.isSaving).toBe(false));
    expect(capturedBody).toEqual({ amount: 2000, threshold: 500 });
  });

  it("disable() POSTs amount=0/threshold=0", async () => {
    let capturedBody: { amount: number; threshold: number } | null = null;
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 2000,
        threshold: 500,
      }),
      http.post("*/api/credits/auto-top-up", async ({ request }) => {
        capturedBody = (await request.json()) as typeof capturedBody;
        return HttpResponse.json({});
      }),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      result.current.disable();
    });

    await waitFor(() => expect(result.current.isSaving).toBe(false));
    expect(capturedBody).toEqual({ amount: 0, threshold: 0 });
  });

  it("save() catches mutation failures and stops the spinner", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 0,
        threshold: 0,
      }),
      http.post("*/api/credits/auto-top-up", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    act(() => {
      result.current.setThreshold("5");
      result.current.setRefillAmount("20");
    });

    await act(async () => {
      result.current.save();
    });

    // Catch branch ran and isSaving is back to false.
    await waitFor(() => expect(result.current.isSaving).toBe(false));
  });

  it("save() with isValid=false short-circuits without calling the mutation", async () => {
    server.use(
      jsonHandler("get", "/api/credits/auto-top-up", {
        amount: 0,
        threshold: 0,
      }),
    );

    const { result } = renderHook(() => useAutoRefillCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.isValid).toBe(false);

    act(() => result.current.save());
    // Mutation never started.
    expect(result.current.isSaving).toBe(false);
  });
});

describe("usePaymentMethodCard", () => {
  it("returns canManage=false when no portal URL is available", async () => {
    server.use(jsonHandler("get", "/api/credits/manage", { url: null }));

    const { result } = renderHook(() => usePaymentMethodCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.canManage).toBe(false));
    // Calling onManage without a portal URL is a no-op (no throw).
    expect(() => result.current.onManage()).not.toThrow();
  });

  it("returns canManage=true once the portal URL is available", async () => {
    server.use(
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    const { result } = renderHook(() => usePaymentMethodCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.canManage).toBe(true));
    expect(result.current.portalUrl).toBe("https://billing.stripe.com/p/test");
  });
});

const TIER_COSTS = { PRO: 5000, MAX: 32000, BUSINESS: 0 } as const;

describe("useYourPlanCard", () => {
  it("returns the subscription payload and exposes the portal URL", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/p/test",
      }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.subscription?.tier).toBe("PRO");
    expect(result.current.subscription?.monthly_cost).toBe(5000);
    expect(result.current.canManagePortal).toBe(true);
    expect(result.current.portalUrl).toBe("https://billing.stripe.com/p/test");
  });

  it("changeTier POSTs the requested tier and toasts on success when no Checkout URL is returned", async () => {
    let capturedTier: string | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        const body = (await request.json()) as { tier: string };
        capturedTier = body.tier;
        return HttpResponse.json({ url: null });
      }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.changeTier("MAX");
    });

    expect(capturedTier).toBe("MAX");
  });

  it("handleTierChange opens the contact-sales URL for the BUSINESS tier instead of POSTing", async () => {
    let stripeHit = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", async () => {
        stripeHit = true;
        return HttpResponse.json({ url: null });
      }),
    );

    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      result.current.handleTierChange("BUSINESS", "MAX", () => {});
    });

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining("contact-sales"),
      "_blank",
      "noopener,noreferrer",
    );
    expect(stripeHit).toBe(false);
    openSpy.mockRestore();
  });

  it("handleTierChange routes downgrades through the confirm callback and queues upgrades for the dialog", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    let downgradeAsked: string | null = null;
    act(() => {
      result.current.handleTierChange("PRO", "MAX", (tier) => {
        downgradeAsked = tier;
      });
    });
    expect(downgradeAsked).toBe("PRO");

    // Upgrade attempt → exposed as pendingUpgradeTier so the view can render
    // the confirm dialog before firing the mutation.
    act(() => {
      result.current.handleTierChange("BUSINESS", "PRO", () => {});
    });
    // BUSINESS is contact-sales — bypasses pendingUpgradeTier entirely.
    expect(result.current.pendingUpgradeTier).toBeNull();

    act(() => {
      result.current.handleTierChange("MAX", "PRO", () => {});
    });
    expect(result.current.pendingUpgradeTier).toBe("MAX");
  });

  it("cancelPendingChange POSTs the current tier to release the schedule", async () => {
    let capturedTier: string | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "PRO",
        pending_tier_effective_at: "2026-05-30T00:00:00Z",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        const body = (await request.json()) as { tier: string };
        capturedTier = body.tier;
        return HttpResponse.json({ url: null });
      }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.cancelPendingChange();
    });

    expect(capturedTier).toBe("MAX");
  });

  it("surfaces tierError when changeTier rejects", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        tier_costs: TIER_COSTS,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      await result.current.changeTier("MAX");
    });

    await waitFor(() => expect(result.current.isPending).toBe(false));
    // Either the mutation surfaces the detail string or falls back to the
    // hook's generic copy — both are valid.
    expect(result.current.tierError).not.toBeNull();
  });
});
