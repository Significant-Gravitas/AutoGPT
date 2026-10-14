import { server } from "@/mocks/mock-server";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
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

// These cover the hosted billing UI, which self-hosted builds hide.
vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
  useBillingStatus: () => ({ isBillingEnabled: true, isResolved: true }),
}));

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

const paymentPage = "https://invoice.stripe.com/i/acct_1/test_pay";

async function openUpgrade() {
  fireEvent.click(
    await screen.findByRole("button", { name: "Upgrade to Max" }),
  );
  return screen.findByRole("dialog", { name: "Start Max today?" });
}

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  toast.mockReset();
  analytics.trackPaywallViewed.mockReset();
  analytics.trackPlanSelected.mockReset();
  vi.restoreAllMocks();
});

describe("upgrading past the trial's plan", () => {
  it.each([
    ["monthly", trialOffer, "$320 / month"],
    ["yearly", yearlyOffer, "$3,264 / year"],
  ] as const)(
    "charges the saved card for the next tier on the %s cycle once confirmed",
    async (cycle, offer, price) => {
      const assign = vi.spyOn(window.location, "assign");
      mockBilling(trialResponse({ ...cancelPending, offer }));
      const body = mockPlanRequest(() =>
        HttpResponse.json({ ...trialSubscription, tier: "MAX", url: "" }),
      );
      render(<SettingsBillingPage />);

      const dialog = await openUpgrade();
      expect(dialog.textContent).toContain(price);
      expect(body).not.toHaveBeenCalled();
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Subscribe to Max" }),
      );

      await waitFor(() =>
        expect(toast).toHaveBeenCalledWith({
          title: "You're on Max",
          description: "Your trial has ended and your plan starts today.",
        }),
      );
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
      expect(assign).not.toHaveBeenCalled();
    },
  );

  it("keeps the trial without a request when the person backs out", async () => {
    mockBilling();
    const body = mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: "" }),
    );
    render(<SettingsBillingPage />);

    const dialog = await openUpgrade();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Keep my trial" }),
    );

    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(body).not.toHaveBeenCalled();
  });

  it("sends the customer to Stripe's page when their bank wants to confirm", async () => {
    const assign = vi.spyOn(window.location, "assign");
    const { hits } = mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: paymentPage }),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      within(await openUpgrade()).getByRole("button", {
        name: "Subscribe to Max",
      }),
    );

    const bank = await screen.findByRole("dialog", {
      name: "Confirm with your bank",
    });
    expect(bank.textContent).toContain("$320 / month payment for Max");
    const pay = within(bank).getByRole("link", { name: "Pay on Stripe" });
    expect(pay.getAttribute("href")).toBe(paymentPage);
    expect(pay.getAttribute("target")).toBe("_blank");
    expect(toast).not.toHaveBeenCalled();
    expect(assign).not.toHaveBeenCalled();

    // Closing re-reads the trial: the customer may have paid in the other tab.
    const trialReads = hits.trial;
    fireEvent.click(within(bank).getByRole("button", { name: "Done" }));
    await waitFor(() =>
      expect(
        screen.queryByRole("dialog", { name: "Confirm with your bank" }),
      ).toBeNull(),
    );
    await waitFor(() => expect(hits.trial).toBeGreaterThan(trialReads));
  });

  it("shows why Max could not start and frees the choices", async () => {
    mockBilling();
    mockPlanRequest(() =>
      HttpResponse.json(
        {
          detail:
            "Your saved card can't be charged. Update it under Payment method, then try again.",
        },
        { status: 409 },
      ),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      within(await openUpgrade()).getByRole("button", {
        name: "Subscribe to Max",
      }),
    );

    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith({
        title: "Unable to start Max",
        description:
          "Your saved card can't be charged. Update it under Payment method, then try again.",
        variant: "destructive",
      }),
    );
    await waitFor(() =>
      expect(
        screen
          .getByRole("button", { name: "Upgrade to Max" })
          .hasAttribute("disabled"),
      ).toBe(false),
    );
  });

  it("does not show user B the payment page user A was sent to", async () => {
    setTrialUser("user-a");
    mockBilling();
    mockPricePerUser();
    const charge = deferredTrialResponse<void>();
    const sent = vi.fn();
    server.use(
      http.post("*/api/credits/subscription", async () => {
        sent();
        await charge.promise;
        return HttpResponse.json({ ...trialSubscription, url: paymentPage });
      }),
    );
    render(<SettingsBillingPage />);
    fireEvent.click(
      within(await openUpgrade()).getByRole("button", {
        name: "Subscribe to Max",
      }),
    );
    await waitFor(() => expect(sent).toHaveBeenCalledOnce());

    act(() => setTrialUser("user-b"));
    await screen.findByText("$30");
    charge.resolve();
    await act(() => new Promise((settle) => setTimeout(settle, 50)));

    expect(
      screen.queryByRole("dialog", { name: "Confirm with your bank" }),
    ).toBeNull();
    expect(toast).not.toHaveBeenCalled();
  });
});
