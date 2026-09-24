import { TooltipProvider } from "@/components/atoms/Tooltip/BaseTooltip";
import { server } from "@/mocks/mock-server";
import { render, waitFor } from "@/tests/integrations/test-utils";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, renderHook } from "@testing-library/react";
import { http, HttpResponse, type JsonBodyType } from "msw";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { usePaymentMethodCard } from "../components/SubscriptionTab/PaymentMethodCard/usePaymentMethodCard";
import { useYourPlanCard } from "../components/SubscriptionTab/YourPlanCard/useYourPlanCard";
import SettingsBillingPage from "../page";

const posthog = vi.hoisted(() => ({
  __loaded: true,
  is_capturing: () => true,
  capture: vi.fn(),
}));
vi.mock("posthog-js", () => ({ default: posthog }));

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

const PORTAL_URL = "https://billing.stripe.com/p/test";
const originalLocation = window.location;

function jsonHandler(method: "get" | "post", path: string, body: JsonBodyType) {
  return http[method](`*${path}`, () => HttpResponse.json(body));
}

function useBillingHandlers(tier = "PRO") {
  server.use(
    jsonHandler("get", "/api/credits/subscription", {
      tier,
      monthly_cost: tier === "MAX" ? 32000 : 5000,
      has_active_stripe_subscription: true,
      status: "active",
    }),
    jsonHandler("get", "/api/credits/manage", { url: PORTAL_URL }),
    jsonHandler("get", "/api/credits", { credits: 1234 }),
    jsonHandler("get", "/api/credits/transactions", {
      transactions: [],
      next_transaction_time: null,
    }),
    jsonHandler("get", "/api/credits/auto-top-up", { amount: 0, threshold: 0 }),
    jsonHandler("get", "/api/credits/invoices", []),
    jsonHandler("get", "/api/v2/chat/copilot-usage", {
      daily: { percent_used: 0, resets_at: null },
      weekly: { percent_used: 0, resets_at: null },
    }),
  );
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

// Opening the portal navigates away; swap in a plain object so the test
// environment survives the assignment.
function stubLocation() {
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: { origin: "https://app.test", href: "https://app.test/" },
  });
}

function eventsNamed(name: string) {
  return posthog.capture.mock.calls.filter(([event]) => event === name);
}

beforeEach(() => {
  posthog.capture.mockClear();
  sessionStorage.clear();
});

afterEach(() => {
  mockSearchParams.current = new URLSearchParams();
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
});

describe("billing page — abandoned Stripe checkouts", () => {
  it.each([
    [{ topup: "cancel" }, "top_up"],
    [{ subscription: "cancelled" }, "subscription"],
    [{ trial: "cancelled" }, "trial"],
  ])("reports %o as checkout_abandoned (%s)", async (params, checkoutKind) => {
    useBillingHandlers();
    mockSearchParams.current = new URLSearchParams(params);

    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(eventsNamed("checkout_abandoned")).toEqual([
        [
          "checkout_abandoned",
          { checkout_kind: checkoutKind, surface: "billing" },
        ],
      ]),
    );
  });

  it("reports nothing on a successful return", async () => {
    useBillingHandlers();
    mockSearchParams.current = new URLSearchParams({ subscription: "success" });

    render(<SettingsBillingPage />);

    await waitFor(() => expect(eventsNamed("paywall_viewed")).toHaveLength(1));
    expect(eventsNamed("checkout_abandoned")).toEqual([]);
  });
});

describe("billing page — plan picker and portal", () => {
  it("reports the billing plan picker once its data loads", async () => {
    useBillingHandlers();

    renderHook(() => useYourPlanCard(), { wrapper: makeWrapper() });

    await waitFor(() =>
      expect(eventsNamed("paywall_viewed")).toEqual([
        ["paywall_viewed", { surface: "billing" }],
      ]),
    );
  });

  it("reports the upgrade target as plan_selected", async () => {
    useBillingHandlers("MAX");
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.isLoading).toBe(false));

    await act(async () => {
      result.current.onUpgrade();
    });

    expect(eventsNamed("plan_selected")).toEqual([
      [
        "plan_selected",
        {
          subscription_tier: "BUSINESS",
          billing_cycle: "monthly",
          surface: "billing",
        },
      ],
    ]);
    openSpy.mockRestore();
  });

  it("reports the billing portal from the plan card", async () => {
    useBillingHandlers();

    const { result } = renderHook(() => useYourPlanCard(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.canManagePortal).toBe(true));
    stubLocation();

    act(() => result.current.onManage());

    expect(eventsNamed("billing_portal_opened")).toEqual([
      ["billing_portal_opened", { surface: "billing" }],
    ]);
    expect(window.location.href).toBe(PORTAL_URL);
  });

  it("reports the billing portal from the payment method card", async () => {
    useBillingHandlers();

    const { result } = renderHook(() => usePaymentMethodCard(), {
      wrapper: makeWrapper(),
    });
    await waitFor(() => expect(result.current.canManage).toBe(true));
    stubLocation();

    act(() => result.current.onManage());

    expect(eventsNamed("billing_portal_opened")).toEqual([
      ["billing_portal_opened", { surface: "billing_payment_method" }],
    ]);
  });
});
