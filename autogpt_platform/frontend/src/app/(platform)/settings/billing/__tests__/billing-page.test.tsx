import { http, HttpResponse, type JsonBodyType } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";

import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";

import { AutomationCreditsTab } from "../components/AutomationCreditsTab/AutomationCreditsTab";
import SettingsBillingPage from "../page";

// Allow per-test override of the search params Next.js exposes to the page.
const mockSearchParams = { current: new URLSearchParams() };
const mockRouterReplace = vi.fn();
vi.mock("next/navigation", async (importOriginal) => {
  const actual = await importOriginal<typeof import("next/navigation")>();
  return {
    ...actual,
    useSearchParams: () => mockSearchParams.current,
    useRouter: () => ({
      push: vi.fn(),
      replace: mockRouterReplace,
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    }),
    usePathname: () => "/settings/billing",
    useParams: () => ({}),
  };
});

afterEach(() => {
  mockSearchParams.current = new URLSearchParams();
  mockRouterReplace.mockReset();
});

const SUBSCRIPTION_RESPONSE = {
  tier: "PRO",
  monthly_cost: 5000,
  has_active_stripe_subscription: true,
  status: "active",
};

const PAYMENT_PORTAL_RESPONSE = { url: "https://billing.stripe.com/p/test" };

const CREDITS_RESPONSE = { credits: 1234 };

const HISTORY_EMPTY = { transactions: [], next_transaction_time: null };

const AUTO_TOP_UP_OFF = { amount: 0, threshold: 0 };

const COPILOT_USAGE_RESPONSE = {
  daily: { percent_used: 12.5, resets_at: null },
  weekly: { percent_used: 30, resets_at: null },
};

const INVOICES_RESPONSE: never[] = [];

function jsonHandler(method: "get" | "post", path: string, body: JsonBodyType) {
  return http[method](`*${path}`, () => HttpResponse.json(body));
}

function useDefaultBillingHandlers() {
  server.use(
    jsonHandler("get", "/api/credits/subscription", SUBSCRIPTION_RESPONSE),
    jsonHandler("get", "/api/credits/manage", PAYMENT_PORTAL_RESPONSE),
    jsonHandler("get", "/api/credits", CREDITS_RESPONSE),
    jsonHandler("get", "/api/credits/transactions", HISTORY_EMPTY),
    jsonHandler("get", "/api/credits/auto-top-up", AUTO_TOP_UP_OFF),
    jsonHandler("get", "/api/credits/invoices", INVOICES_RESPONSE),
    jsonHandler("get", "/api/v2/chat/copilot-usage", COPILOT_USAGE_RESPONSE),
  );
}

describe("Settings billing page (integration)", () => {
  it("renders the Subscription tab with plan + payment + invoices once data resolves", async () => {
    useDefaultBillingHandlers();
    render(<SettingsBillingPage />);

    expect(screen.getByRole("heading", { name: "Billing" })).toBeDefined();

    const subscriptionTab = screen.getByRole("tab", { name: "Subscription" });
    expect(subscriptionTab.getAttribute("data-state")).toBe("active");

    expect(await screen.findByText("Your plan")).toBeDefined();
    expect(await screen.findByText("Pro")).toBeDefined();
    expect(screen.getByText(/\$50\.00 \/ month/)).toBeDefined();

    // Invoices card empty-state copy (no invoices fixture above).
    expect(await screen.findByText(/No invoices yet/i)).toBeDefined();
  });

  it("AutomationCreditsTab renders the balance + add-credits CTA from API data", async () => {
    useDefaultBillingHandlers();
    render(<AutomationCreditsTab />);

    // Balance comes from /api/credits → 1234 cents → $12.34 (Intl-formatted).
    expect(await screen.findByText("$12.34")).toBeDefined();
    expect(screen.getByText("Automation credits")).toBeDefined();
    expect(screen.getByRole("button", { name: /add credits/i })).toBeDefined();
  });

  it("clears the ?topup=success query and shows a success toast after a Stripe redirect", async () => {
    useDefaultBillingHandlers();
    server.use(
      http.patch("*/api/credits/fulfill_checkout", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
    );
    mockSearchParams.current = new URLSearchParams({ topup: "success" });

    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
  });

  it("clears the ?topup=cancel query without firing fulfillCheckout", async () => {
    useDefaultBillingHandlers();
    let fulfillCalled = false;
    server.use(
      http.patch("*/api/credits/fulfill_checkout", () => {
        fulfillCalled = true;
        return HttpResponse.json({}, { status: 200 });
      }),
    );
    mockSearchParams.current = new URLSearchParams({ topup: "cancel" });

    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
    // Cancel branch must NOT fulfill — webhook is the source of truth and
    // the user explicitly aborted the checkout.
    expect(fulfillCalled).toBe(false);
  });

  it("clears the ?subscription=success query after a Stripe redirect", async () => {
    useDefaultBillingHandlers();
    mockSearchParams.current = new URLSearchParams({ subscription: "success" });

    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
  });

  it("clears the ?subscription=cancelled query after the user aborts checkout", async () => {
    useDefaultBillingHandlers();
    mockSearchParams.current = new URLSearchParams({
      subscription: "cancelled",
    });

    render(<SettingsBillingPage />);

    await waitFor(() =>
      expect(mockRouterReplace).toHaveBeenCalledWith("/settings/billing"),
    );
  });

  it("activates the Automation Credits tab when a Stripe topup redirect lands on the page", async () => {
    useDefaultBillingHandlers();
    server.use(
      http.patch("*/api/credits/fulfill_checkout", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
    );
    // Topup flow originates from the Automation Credits tab; the redirect
    // back from Stripe must NOT bounce the user to Subscription.
    mockSearchParams.current = new URLSearchParams({ topup: "success" });

    render(<SettingsBillingPage />);

    const automationCreditsTab = await screen.findByRole("tab", {
      name: /automation credits/i,
    });
    expect(automationCreditsTab.getAttribute("data-state")).toBe("active");
  });

  it("activates the requested tab when ?tab=automation-credits is present (deep-link)", async () => {
    useDefaultBillingHandlers();
    mockSearchParams.current = new URLSearchParams({
      tab: "automation-credits",
    });

    render(<SettingsBillingPage />);

    const automationCreditsTab = await screen.findByRole("tab", {
      name: /automation credits/i,
    });
    expect(automationCreditsTab.getAttribute("data-state")).toBe("active");
  });

  it("ignores an unknown ?tab= value and falls back to the Subscription default", async () => {
    useDefaultBillingHandlers();
    mockSearchParams.current = new URLSearchParams({ tab: "bogus" });

    render(<SettingsBillingPage />);

    const subscriptionTab = await screen.findByRole("tab", {
      name: "Subscription",
    });
    expect(subscriptionTab.getAttribute("data-state")).toBe("active");
  });

  it("AutomationCreditsTab renders the ErrorCard when the balance fetch fails", async () => {
    server.use(
      jsonHandler("get", "/api/credits/subscription", SUBSCRIPTION_RESPONSE),
      jsonHandler("get", "/api/credits/manage", PAYMENT_PORTAL_RESPONSE),
      jsonHandler("get", "/api/credits/transactions", HISTORY_EMPTY),
      jsonHandler("get", "/api/credits/auto-top-up", AUTO_TOP_UP_OFF),
      jsonHandler("get", "/api/credits/invoices", INVOICES_RESPONSE),
      jsonHandler("get", "/api/v2/chat/copilot-usage", COPILOT_USAGE_RESPONSE),
      http.get("*/api/credits", () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<AutomationCreditsTab />);

    await waitFor(() => {
      // ErrorCard renders instead of the silent "$0.00" misread.
      expect(screen.queryByText("$0.00")).toBeNull();
    });
  });
});
