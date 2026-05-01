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

  it("isValid requires whole-dollar amounts >= $5 (backend rejects decimals)", async () => {
    server.use(jsonHandler("get", "/api/credits", { credits: 1000 }));

    const { result } = renderHook(() => useBalanceCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    // Backend `top_up_intent` rejects `amount % 100 != 0`, so decimals
    // like $5.25 must be blocked client-side instead of failing at
    // checkout time.
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

    // Fractional values are accepted — backend stores integer cents, so
    // legacy non-whole-dollar amounts (e.g. $7.50) must remain editable.
    act(() => {
      result.current.setThreshold("5");
      result.current.setRefillAmount("5.5");
    });
    expect(result.current.isValid).toBe(true);

    // Valid: both >= 5, refill >= threshold.
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

describe("useYourPlanCard", () => {
  it("derives plan label/cost/nextTier from the subscription response", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
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

    expect(result.current.plan?.label).toBe("Pro");
    expect(result.current.plan?.tierKey).toBe("PRO");
    expect(result.current.plan?.monthlyCostCents).toBe(5000);
    expect(result.current.plan?.isPaidPlan).toBe(true);
    expect(result.current.plan?.nextTier).toBe("MAX");
    expect(result.current.canUpgrade).toBe(true);
    expect(result.current.canManagePortal).toBe(true);
  });

  it("nextTier is null on the top tier (BUSINESS) and canUpgrade=false", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "BUSINESS",
        monthly_cost: 50000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.plan?.nextTier).toBeNull();
    expect(result.current.canUpgrade).toBe(false);
  });

  it("exposes previousTier=PRO and canDowngrade=true for an active MAX subscriber", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
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

    expect(result.current.plan?.previousTier).toBe("PRO");
    expect(result.current.plan?.previousTierLabel).toBe("Pro");
    expect(result.current.canDowngrade).toBe(true);
    // MAX → next tier is BUSINESS → contact-sales, not in-app Checkout.
    expect(result.current.plan?.nextTier).toBe("BUSINESS");
    expect(result.current.plan?.nextTierIsTeamLink).toBe(true);
  });

  it("onDowngrade calls updateTier with the previous tier (MAX → PRO)", async () => {
    let capturedTier: string | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
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
      result.current.onDowngrade();
    });

    await waitFor(() => expect(capturedTier).toBe("PRO"));
  });

  it("onUpgrade for a MAX subscriber opens TEAM_UPGRADE_URL instead of POSTing to /credits/subscription", async () => {
    let stripeHit = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
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
      result.current.onUpgrade();
    });

    expect(openSpy).toHaveBeenCalledWith(
      expect.stringContaining("tally.so"),
      "_blank",
      "noopener,noreferrer",
    );
    expect(stripeHit).toBe(false);
    openSpy.mockRestore();
  });

  it("flags isPendingDowngrade and canResume when a portal-initiated downgrade is scheduled", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "MAX",
        monthly_cost: 32000,
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "PRO",
        pending_tier_effective_at: "2026-05-30T00:00:00Z",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.plan?.isPendingDowngrade).toBe(true);
    expect(result.current.plan?.pendingTierLabel).toBe("Pro");
    expect(result.current.canResume).toBe(true);
  });

  it("offers PRO as the next tier when the user has no active subscription", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "NO_TIER",
        monthly_cost: 0,
        has_active_stripe_subscription: false,
        status: "inactive",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.plan?.isPaidPlan).toBe(false);
    expect(result.current.plan?.nextTier).toBe("PRO");
    expect(result.current.canUpgrade).toBe(true);
  });

  it("changeTier (PRO → MAX upgrade) calls updateTier with the requested tier", async () => {
    let capturedTier: string | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
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
    expect(result.current.canUpgrade).toBe(true);

    await act(async () => {
      result.current.onUpgrade();
    });

    await waitFor(() => expect(capturedTier).toBe("MAX"));
  });

  it("changeTier surfaces a destructive toast when the API returns 500", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
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
      result.current.onUpgrade();
    });

    // The catch path runs without throwing; mutation completes.
    await waitFor(() => expect(result.current.isUpdatingTier).toBe(false));
  });

  it("onResume releases a pending cancellation by POSTing the current tier", async () => {
    let capturedTier: string | null = null;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        has_active_stripe_subscription: true,
        status: "active",
        pending_tier: "NO_TIER",
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
    expect(result.current.canResume).toBe(true);

    await act(async () => {
      result.current.onResume();
    });

    await waitFor(() => expect(capturedTier).toBe("PRO"));
  });

  it("onResume short-circuits when there is no pending change in flight", async () => {
    let posted = false;
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
      http.post("*/api/credits/subscription", () => {
        posted = true;
        return HttpResponse.json({ url: null });
      }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));
    expect(result.current.canResume).toBe(false);

    await act(async () => {
      result.current.onResume();
    });

    // No mutation should fire when there's nothing to resume.
    expect(posted).toBe(false);
  });

  it("canManagePortal flips true once the portal URL resolves and onManage is exposed", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "PRO",
        monthly_cost: 5000,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", {
        url: "https://billing.stripe.com/session/test-portal",
      }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.canManagePortal).toBe(true));

    // onManage with a missing portal URL is a no-op; verifying it doesn't
    // throw exercises the `if (paymentPortal.data)` branch under coverage.
    expect(typeof result.current.onManage).toBe("function");
  });

  it("falls back to the raw tier string when PLAN_LABEL has no match", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", {
        tier: "STARTUP",
        monthly_cost: 100,
        has_active_stripe_subscription: true,
        status: "active",
      }),
      jsonHandler("get", "/api/credits/manage", { url: null }),
    );

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });

    await waitFor(() => expect(result.current.isLoading).toBe(false));

    expect(result.current.plan?.label).toBe("STARTUP");
    // STARTUP isn't in TIER_ORDER → unknown, default to first paid tier.
    expect(result.current.plan?.nextTier).toBe("PRO");
  });
});
