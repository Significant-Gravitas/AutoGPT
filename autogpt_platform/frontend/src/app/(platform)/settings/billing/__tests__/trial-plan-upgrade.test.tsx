import { server } from "@/mocks/mock-server";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  deferredTrialResponse,
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import {
  cancelPending,
  mockBilling,
  mockPlanRequest,
  mockPricePerUser,
  returnURLs,
  trialSubscription,
  yearlyOffer,
} from "./trial-plan-fixtures";

const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  toast,
}));

const analytics = vi.hoisted(() => ({
  trackPaywallViewed: vi.fn(),
  trackPlanSelected: vi.fn(),
}));
vi.mock(
  "@/services/analytics/monetization-analytics",
  async (importOriginal) => ({
    ...(await importOriginal<
      typeof import("@/services/analytics/monetization-analytics")
    >()),
    ...analytics,
  }),
);

const checkoutURL = "https://checkout.stripe.com/c/pay/cs_max";

function stubNavigation() {
  return vi.spyOn(window.location, "assign").mockImplementation(() => {});
}

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  toast.mockReset();
  analytics.trackPaywallViewed.mockReset();
  analytics.trackPlanSelected.mockReset();
  removeGtagShim();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("upgrading past the trial's plan", () => {
  it.each([
    ["monthly", trialOffer],
    ["yearly", yearlyOffer],
  ] as const)(
    "opens Stripe Checkout for the next tier on the %s cycle",
    async (cycle, offer) => {
      const assign = stubNavigation();
      mockBilling(trialResponse({ ...cancelPending, offer }));
      const body = mockPlanRequest(() =>
        HttpResponse.json({ ...trialSubscription, url: checkoutURL }),
      );
      render(<SettingsBillingPage />);
      fireEvent.click(
        await screen.findByRole("button", { name: "Upgrade to Max" }),
      );

      await waitFor(() => expect(assign).toHaveBeenCalledWith(checkoutURL));
      expect(body).toHaveBeenCalledWith({
        tier: "MAX",
        billing_cycle: cycle,
        surface: "billing",
        ...returnURLs("MAX", cycle),
      });
      expect(analytics.trackPlanSelected).toHaveBeenCalledWith({
        subscription_tier: "MAX",
        billing_cycle: cycle,
        surface: "billing",
      });
      expect(screen.queryByRole("dialog")).toBeNull();
      expect(toast).not.toHaveBeenCalled();
    },
  );

  it("reports begin_checkout at the next tier's price before leaving", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
    const gtagCalls = installGtagShim();
    const assign = stubNavigation();
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: checkoutURL }),
    );
    render(<SettingsBillingPage />);
    const upgrade = await screen.findByRole("button", {
      name: "Upgrade to Max",
    });
    fireEvent.click(upgrade);

    await waitFor(() =>
      expect(gtagCalls).toContainEqual([
        "event",
        "conversion",
        {
          send_to: "AW-123/BC",
          value: 320,
          currency: "USD",
          event_callback: expect.any(Function),
        },
      ]),
    );
    expect(upgrade.hasAttribute("disabled")).toBe(true);
    await waitFor(() => expect(assign).toHaveBeenCalledWith(checkoutURL));
  });

  it("frees the choices after handing off, so a page restored from history works", async () => {
    const assign = stubNavigation();
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: checkoutURL }),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    );

    await waitFor(() => expect(assign).toHaveBeenCalledOnce());
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Upgrade to Max" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(
      screen
        .getByRole("button", { name: "Subscribe to Pro" })
        .hasAttribute("disabled"),
    ).toBe(false);
  });

  it("shows why Checkout could not start", async () => {
    const assign = stubNavigation();
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json(
        { detail: "Stripe is unavailable. Please retry." },
        { status: 502 },
      ),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    );
    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Stripe is unavailable. Please retry.");
    expect(assign).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Upgrade to Max" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(screen.getByRole("alert").textContent).toContain(
      "Stripe is unavailable. Please retry.",
    );
  });

  it("does not send user B to the Checkout user A started", async () => {
    const assign = stubNavigation();
    setTrialUser("user-a");
    mockBilling();
    mockPricePerUser();
    const checkout = deferredTrialResponse<void>();
    const sent = vi.fn();
    server.use(
      http.post("*/api/credits/subscription", async () => {
        sent();
        await checkout.promise;
        return HttpResponse.json({
          ...trialSubscription,
          url: "https://checkout.stripe.com/c/pay/cs_user_a",
        });
      }),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Upgrade to Max" }),
    );
    await waitFor(() => expect(sent).toHaveBeenCalledOnce());

    act(() => setTrialUser("user-b"));
    await screen.findByText("$30");
    checkout.resolve();
    await act(() => new Promise((settle) => setTimeout(settle, 50)));

    expect(assign).not.toHaveBeenCalled();
    expect(screen.queryByRole("alert")).toBeNull();
  });
});
