import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { TrialCard } from "@/components/organisms/TrialCard/TrialCard";
import { server } from "@/mocks/mock-server";
import {
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

beforeEach(() => setTrialUser());
afterEach(() => {
  setTrialUser(null);
  vi.restoreAllMocks();
});

describe("trial billing actions", () => {
  it("shows trial usage and cancels without ending the remaining trial access", async () => {
    const pending = deferredTrialResponse<ReturnType<typeof trialResponse>>();
    const cancel = vi.fn(() => pending.promise);
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
      getPostTrialsCancelTrialMockHandler200(cancel),
    );
    render(<TrialCard />);
    const button = await screen.findByRole("button", { name: "Cancel trial" });
    expect(screen.getByRole("progressbar").getAttribute("value")).toBe("42.4");
    expect(screen.getByText(/allowance used: 42%/i)).toBeDefined();
    fireEvent.click(button);
    await waitFor(() => expect(cancel).toHaveBeenCalledOnce());
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    pending.resolve(trialResponse({ cancel_at_period_end: true }));
    expect(await screen.findByText(/Cancellation confirmed/)).toBeDefined();
    expect(screen.getByText("Your trial")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
    expect(cancel).toHaveBeenCalledOnce();
  });

  it("keeps cancellation retryable when the request fails", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(trialResponse()),
      http.post("*/api/credits/trial/cancel", () =>
        HttpResponse.json({ detail: "Stripe is unavailable" }, { status: 502 }),
      ),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Cancel trial" }),
    );
    expect(await screen.findByRole("alert")).toBeDefined();
    expect(
      screen
        .getByRole("button", { name: "Cancel trial" })
        .hasAttribute("disabled"),
    ).toBe(false);
    expect(screen.queryByText(/Cancellation confirmed/)).toBeNull();
    server.use(
      getPostTrialsCancelTrialMockHandler200(
        trialResponse({ cancel_at_period_end: true }),
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Cancel trial" }));
    await screen.findByText(/Cancellation confirmed/);
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it.each(["billing", "onboarding"] as const)(
    "sends the accepted token and %s destination before redirecting",
    async (returnTo) => {
      const assign = vi
        .spyOn(window.location, "assign")
        .mockImplementation(() => {});
      const body = vi.fn();
      server.use(
        getGetTrialsGetTrialStatusMockHandler200(
          trialResponse({ eligible: true, active: false, status: null }),
        ),
        getPostTrialsStartTrialCheckoutMockHandler200(async ({ request }) => {
          body(await request.json());
          return { url: "https://checkout.stripe.com/c/pay/test_trial" };
        }),
      );
      render(<TrialCard returnTo={returnTo} />);
      fireEvent.click(
        await screen.findByRole("button", { name: /start 7-day trial/i }),
      );
      await waitFor(() =>
        expect(assign).toHaveBeenCalledWith(
          "https://checkout.stripe.com/c/pay/test_trial",
        ),
      );
      expect(body).toHaveBeenCalledWith({
        offer_token: trialOffer.token,
        return_to: returnTo,
      });
    },
  );

  it("refreshes an expired offer after checkout is rejected", async () => {
    let rejected = false;
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(() =>
        trialResponse({
          eligible: true,
          active: false,
          status: null,
          offer: rejected
            ? { ...trialOffer, duration_days: 10, token: "b".repeat(64) }
            : trialOffer,
        }),
      ),
      http.post("*/api/credits/trial", () => {
        rejected = true;
        return HttpResponse.json(
          { detail: "Refresh the offer" },
          { status: 409 },
        );
      }),
    );
    render(<TrialCard />);
    fireEvent.click(
      await screen.findByRole("button", { name: /start 7-day trial/i }),
    );
    await screen.findByRole("alert");
    expect(
      (
        await screen.findByRole("button", { name: /start 10-day trial/i })
      ).hasAttribute("disabled"),
    ).toBe(false);
  });

  it("does not offer cancellation or imply paid access after expiry", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({ active: false, status: "past_due", ends_at: null }),
      ),
    );
    render(<TrialCard />);
    await screen.findByText("Your trial has ended");
    expect(
      screen.getByText(/Paid access requires a successful payment/),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
    expect(screen.queryByRole("progressbar")).toBeNull();
  });

  it("recovers a failed status request through the visible retry action", async () => {
    server.use(
      http.get("*/api/credits/trial", () =>
        HttpResponse.json({ detail: "Unavailable" }, { status: 503 }),
      ),
    );
    render(<TrialCard />);
    const retry = await screen.findByRole("button", { name: /try again/i });
    server.use(getGetTrialsGetTrialStatusMockHandler200(trialResponse()));
    fireEvent.click(retry);
    await screen.findByRole("button", { name: "Cancel trial" });
    expect(screen.queryByRole("button", { name: /try again/i })).toBeNull();
  });

  it("renders a tunable yearly Team offer in a zero-decimal currency", async () => {
    server.use(
      getGetTrialsGetTrialStatusMockHandler200(
        trialResponse({
          eligible: true,
          active: false,
          offer: {
            ...trialOffer,
            tier: "BUSINESS",
            billing_cycle: "yearly",
            currency: "jpy",
            unit_amount: 1000,
          },
        }),
      ),
    );
    render(<TrialCard />);
    await screen.findByText(/Try AutoGPT Team for 7 days/);
    expect(screen.getByText(/¥1,000 \/ year/)).toBeDefined();
  });
});
