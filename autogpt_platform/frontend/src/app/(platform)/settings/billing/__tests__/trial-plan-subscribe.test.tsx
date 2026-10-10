import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { HttpResponse } from "msw";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import SettingsBillingPage from "../page";
import {
  cancelPending,
  mockBilling,
  mockPlanRequest,
  mockPricePerUser,
  openConfirmation,
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

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  toast.mockReset();
  analytics.trackPaywallViewed.mockReset();
  analytics.trackPlanSelected.mockReset();
  vi.restoreAllMocks();
});

describe("subscribing to the trial's plan", () => {
  it("charges today only after confirming, then refreshes trial and plan", async () => {
    const { state, hits } = mockBilling();
    const body = mockPlanRequest(() => {
      state.trial = trialResponse({ ...cancelPending, converted: true });
      state.subscription = { ...trialSubscription, tier: "PRO" };
      return HttpResponse.json({ ...state.subscription, url: "" });
    });
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    expect(
      within(dialog).getByText(
        "Your trial ends now and your saved card is charged $20 / month, plus applicable tax.",
      ),
    ).toBeDefined();
    expect(body).not.toHaveBeenCalled();
    expect(analytics.trackPlanSelected).toHaveBeenCalledWith({
      subscription_tier: "PRO",
      billing_cycle: "monthly",
      surface: "billing",
    });
    const before = { ...hits };

    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );

    await waitFor(() => expect(body).toHaveBeenCalledOnce());
    expect(body).toHaveBeenCalledWith({
      tier: "PRO",
      billing_cycle: "monthly",
      surface: "billing",
      ...returnURLs("PRO", "monthly"),
    });
    await waitFor(() => expect(hits.trial).toBeGreaterThan(before.trial));
    await waitFor(() =>
      expect(hits.subscription).toBeGreaterThan(before.subscription),
    );
    await waitFor(() => expect(hits.invoices).toBeGreaterThan(before.invoices));
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({ title: "You're on Pro" }),
    );
    await waitFor(() =>
      expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull(),
    );
  });

  it("sends the trial's own cycle for a yearly trial", async () => {
    mockBilling(trialResponse({ ...cancelPending, offer: yearlyOffer }));
    const body = mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: "" }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    expect(
      within(dialog).getByText(
        "Your trial ends now and your saved card is charged $510 / year, plus applicable tax.",
      ),
    ).toBeDefined();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );
    await waitFor(() =>
      expect(body).toHaveBeenCalledWith({
        tier: "PRO",
        billing_cycle: "yearly",
        surface: "billing",
        ...returnURLs("PRO", "yearly"),
      }),
    );
  });

  it("keeps the cents of a price that has them", async () => {
    mockBilling(
      trialResponse({
        ...cancelPending,
        offer: { ...yearlyOffer, billing_cycle: "monthly", unit_amount: 1999 },
      }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    expect(
      within(dialog).getByText(
        "Your trial ends now and your saved card is charged $19.99 / month, plus applicable tax.",
      ),
    ).toBeDefined();
  });

  it("keeps the trial without a request when the person backs out", async () => {
    mockBilling();
    const body = mockPlanRequest(() =>
      HttpResponse.json({ ...trialSubscription, url: "" }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Keep my trial" }),
    );
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(body).not.toHaveBeenCalled();
    expect(
      screen.getByRole("button", { name: "Subscribe to Pro" }),
    ).toBeDefined();
  });

  it("shows why the charge failed and keeps the choices usable", async () => {
    mockBilling();
    const body = mockPlanRequest(() =>
      HttpResponse.json({ detail: "Your card was declined." }, { status: 402 }),
    );
    render(<SettingsBillingPage />);
    const dialog = await openConfirmation();
    fireEvent.click(
      within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
    );

    const alert = await screen.findByRole("alert");
    expect(alert.textContent).toContain("Your card was declined.");
    expect(body).toHaveBeenCalledOnce();
    await waitFor(() => expect(screen.queryByRole("dialog")).toBeNull());
    expect(toast).not.toHaveBeenCalled();
    expect(
      screen
        .getByRole("button", { name: "Subscribe to Pro" })
        .hasAttribute("disabled"),
    ).toBe(false);
  });

  it.each([409, 402, 422, 502])(
    "refreshes the trial and plan after a %i refusal",
    async (status) => {
      const { state, hits } = mockBilling();
      mockPlanRequest(() => {
        state.trial = trialResponse({ cancel_keeps_access: true });
        return HttpResponse.json(
          { detail: "Your accepted plan starts after your trial." },
          { status },
        );
      });
      render(<SettingsBillingPage />);
      const dialog = await openConfirmation();
      const before = { ...hits };
      fireEvent.click(
        within(dialog).getByRole("button", { name: "Subscribe to Pro" }),
      );

      await waitFor(() => expect(hits.trial).toBeGreaterThan(before.trial));
      await waitFor(() =>
        expect(hits.subscription).toBeGreaterThan(before.subscription),
      );
      expect(
        await screen.findByRole("button", { name: "Cancel trial" }),
      ).toBeDefined();
      expect(screen.queryByRole("region", { name: "Plan choices" })).toBeNull();
      expect(toast).not.toHaveBeenCalled();
    },
  );

  it("does not show user B the confirmation user A opened", async () => {
    setTrialUser("user-a");
    mockBilling();
    mockPricePerUser();
    render(<SettingsBillingPage />);
    await screen.findByText("$20");
    act(() => setTrialUser("user-b"));
    await screen.findByText("$30");
    act(() => setTrialUser("user-a"));
    await screen.findByText("$20");

    await openConfirmation();
    act(() => setTrialUser("user-b"));

    await screen.findByText("$30");
    expect(screen.queryByRole("dialog")).toBeNull();
  });
});
