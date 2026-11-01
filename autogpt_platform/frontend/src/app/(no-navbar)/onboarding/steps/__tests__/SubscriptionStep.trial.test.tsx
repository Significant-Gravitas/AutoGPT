import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import { removeGtagShim } from "@/tests/integrations/gtag-shim";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  setTrialUser,
  trialOffer,
  trialResponse,
} from "@/tests/integrations/trial-fixtures";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useOnboardingWizardStore } from "../../store";
import { SubscriptionStep } from "../SubscriptionStep/SubscriptionStep";
import { translateText } from "@/tests/integrations/translation";

const posthog = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("@posthog/react", () => ({
  usePostHog: () => posthog,
}));

const originalLocation = window.location;
let checkoutLocation = {
  origin: "http://localhost",
  href: "http://localhost/",
  assign: vi.fn(),
};

beforeEach(() => {
  setTrialUser();
  posthog.capture.mockClear();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().goToStep(3);
  vi.spyOn(environment, "isLocal").mockReturnValue(false);
  checkoutLocation = {
    origin: "http://localhost",
    href: "http://localhost/",
    assign: vi.fn(),
  };
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: checkoutLocation,
  });
});

afterEach(() => {
  setTrialUser(null);
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
  removeGtagShim();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

function eligible(offer = trialOffer) {
  return trialResponse({ eligible: true, active: false, status: null, offer });
}

function mockStatus(response = eligible()) {
  server.use(getGetTrialsGetTrialStatusMockHandler200(response));
}

test.each(["paid", "trial"])(
  "keeps a translated %s checkout usable while the request is pending",
  async (mode) => {
    mockStatus(
      mode === "trial"
        ? eligible()
        : trialResponse({
            eligible: false,
            active: false,
            status: null,
            offer: null,
          }),
    );
    let finishCheckout = () => {};
    const pending = new Promise<void>((resolve) => {
      finishCheckout = resolve;
    });
    const requested = vi.fn();
    server.use(
      http.post(
        `*/api/credits/${mode === "paid" ? "subscription" : "trial"}`,
        async ({ request }) => {
          requested(await request.json());
          await pending;
          return HttpResponse.json({
            url: "https://checkout.stripe.com/translated",
          });
        },
      ),
    );
    render(<SubscriptionStep />);
    const label =
      mode === "paid" ? "Continue to checkout" : "Start my free trial";
    const button = await screen.findByRole("button", { name: label });
    const translatedLabel =
      mode === "paid" ? "تابع إلى الدفع" : "ابدأ الفترة التجريبية";
    const translation = translateText(button, label, translatedLabel);

    try {
      fireEvent.click(button);
      await waitFor(() => expect(requested).toHaveBeenCalledOnce());
      expect(screen.getByRole("button", { name: translatedLabel })).toBe(
        button,
      );
      expect(button.contains(translation)).toBe(true);
      expect(button.hasAttribute("disabled")).toBe(true);
      expect(requested).toHaveBeenCalledWith(
        mode === "paid"
          ? expect.objectContaining({ tier: "PRO", billing_cycle: "monthly" })
          : { offer_token: trialOffer.token, return_to: "onboarding" },
      );
    } finally {
      finishCheckout();
    }

    await waitFor(() => {
      if (mode === "paid")
        expect(checkoutLocation.href).toBe(
          "https://checkout.stripe.com/translated",
        );
      else
        expect(checkoutLocation.assign).toHaveBeenCalledWith(
          "https://checkout.stripe.com/translated",
        );
    });
  },
);

test.each(["paid", "trial"])(
  "can retry a translated %s checkout after a failed request",
  async (mode) => {
    mockStatus(
      mode === "trial"
        ? eligible()
        : trialResponse({
            eligible: false,
            active: false,
            status: null,
            offer: null,
          }),
    );
    let attempts = 0;
    server.use(
      http.post(
        `*/api/credits/${mode === "paid" ? "subscription" : "trial"}`,
        () => {
          attempts++;
          return attempts === 1
            ? HttpResponse.json({ detail: "Please try again" }, { status: 503 })
            : HttpResponse.json({ url: "https://checkout.stripe.com/retried" });
        },
      ),
    );
    render(<SubscriptionStep />);
    const label =
      mode === "paid" ? "Continue to checkout" : "Start my free trial";
    const button = await screen.findByRole("button", { name: label });
    const translation = translateText(button, label, "Translated checkout");

    fireEvent.click(button);
    await waitFor(() => {
      expect(attempts).toBe(1);
      expect(button.hasAttribute("disabled")).toBe(false);
    });
    expect(screen.getByRole("button", { name: "Translated checkout" })).toBe(
      button,
    );
    expect(button.contains(translation)).toBe(true);
    fireEvent.click(button);

    await waitFor(() => {
      if (mode === "paid")
        expect(checkoutLocation.href).toBe(
          "https://checkout.stripe.com/retried",
        );
      else
        expect(checkoutLocation.assign).toHaveBeenCalledWith(
          "https://checkout.stripe.com/retried",
        );
    });
    expect(attempts).toBe(2);
  },
);

test.each(["PRO", "MAX"] as const)(
  "starts the %s trial with the accepted token and onboarding destination",
  async (tier) => {
    mockStatus(eligible({ ...trialOffer, tier }));
    const requestBody = vi.fn();
    server.use(
      getPostTrialsStartTrialCheckoutMockHandler200(async ({ request }) => {
        requestBody(await request.json());
        return { url: "https://checkout.stripe.com/trial" };
      }),
    );
    render(<SubscriptionStep />);
    const start = await screen.findByRole("button", {
      name: "Start my free trial",
    });
    // The server picks the tier; this step never names a plan.
    expect(screen.queryByText(/\bMax\b/)).toBeNull();
    expect(screen.queryByText(/\bTeam\b/)).toBeNull();
    fireEvent.click(start);
    await waitFor(() =>
      expect(checkoutLocation.assign).toHaveBeenCalledWith(
        "https://checkout.stripe.com/trial",
      ),
    );
    expect(requestBody).toHaveBeenCalledExactlyOnceWith({
      offer_token: trialOffer.token,
      return_to: "onboarding",
    });
    // The backend sends checkout_started (checkout_kind: trial) once the
    // Checkout session exists, so the browser sends no checkout event.
    expect(posthog.capture).not.toHaveBeenCalledWith(
      "subscription_trial_checkout_started",
      expect.anything(),
    );
    expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
  },
);

test("re-arms trial abandonment tracking when a new trial checkout starts", async () => {
  // Left by an earlier abandonment in this tab: without clearing it, backing
  // out of this checkout too would go unreported.
  sessionStorage.setItem("posthog_trial_checkout_abandoned_onboarding", "1");
  mockStatus();
  server.use(
    getPostTrialsStartTrialCheckoutMockHandler200({
      url: "https://checkout.stripe.com/trial",
    }),
  );
  render(<SubscriptionStep />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Start my free trial" }),
  );

  await waitFor(() => expect(checkoutLocation.assign).toHaveBeenCalled());
  expect(
    sessionStorage.getItem("posthog_trial_checkout_abandoned_onboarding"),
  ).toBeNull();
});

test("uses the server's duration, currency, renewal cycle, and terms", async () => {
  mockStatus(
    eligible({
      ...trialOffer,
      tier: "MAX",
      duration_days: 14,
      currency: "jpy",
      unit_amount: 1234,
      billing_cycle: "yearly",
    }),
  );
  render(<SubscriptionStep />);
  expect(
    await screen.findByRole("heading", {
      name: "Your first 14 days are on us",
    }),
  ).toBeDefined();
  const timeline = within(
    screen.getByRole("list", { name: "How your trial works" }),
  );
  expect(timeline.getByText("Your first payment: ¥1,234")).toBeDefined();
  expect(
    timeline.getByText(/^Then ¥1,234 a year, plus any applicable tax\./),
  ).toBeDefined();
  expect(screen.getByText(/Stripe will ask for a card/)).toBeDefined();
});

test("shows the trial timeline: today, the reminder, then the first payment", async () => {
  mockStatus();
  render(<SubscriptionStep />);
  expect(
    await screen.findByRole("heading", { name: "Your first 7 days are on us" }),
  ).toBeDefined();
  const steps = within(
    screen.getByRole("list", { name: "How your trial works" }),
  ).getAllByRole("listitem");
  expect(steps.map((step) => step.textContent)).toEqual([
    expect.stringContaining("Your free trial starts"),
    expect.stringContaining("We email you a reminder"),
    expect.stringContaining("Your first payment: $20"),
  ]);
  expect(steps[0].textContent).toContain("Today");
  expect(steps[2].textContent).toContain("plus any applicable tax");
});

test("refreshes a rejected offer and submits its new token on retry", async () => {
  let rejected = false;
  const requestBody = vi.fn();
  server.use(
    getGetTrialsGetTrialStatusMockHandler200(() =>
      eligible(
        rejected
          ? { ...trialOffer, token: "b".repeat(64), duration_days: 10 }
          : trialOffer,
      ),
    ),
    http.post("*/api/credits/trial", async ({ request }) => {
      requestBody(await request.json());
      if (rejected)
        return HttpResponse.json({
          url: "https://checkout.stripe.com/refreshed",
        });
      rejected = true;
      return HttpResponse.json(
        { detail: "Refresh the offer" },
        { status: 409 },
      );
    }),
  );
  render(<SubscriptionStep />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Start my free trial" }),
  );
  expect(await screen.findByRole("alert")).toBeDefined();
  // The refreshed offer replaces the old one on screen.
  await screen.findByRole("heading", { name: "Your first 10 days are on us" });
  fireEvent.click(screen.getByRole("button", { name: "Start my free trial" }));
  await waitFor(() =>
    expect(checkoutLocation.assign).toHaveBeenCalledWith(
      "https://checkout.stripe.com/refreshed",
    ),
  );
  expect(requestBody).toHaveBeenLastCalledWith({
    offer_token: "b".repeat(64),
    return_to: "onboarding",
  });
});

test("preserves active-trial status and confirmed cancellation", async () => {
  mockStatus(trialResponse());
  const cancel = vi.fn(() =>
    trialResponse({ active: false, status: "canceled" }),
  );
  server.use(getPostTrialsCancelTrialMockHandler200(cancel));
  render(<SubscriptionStep />);
  fireEvent.click(await screen.findByRole("button", { name: "Cancel trial" }));
  expect(
    screen.queryByRole("button", { name: "Start my free trial" }),
  ).toBeNull();
  expect(cancel).not.toHaveBeenCalled();
  fireEvent.click(
    within(
      await screen.findByRole("dialog", { name: "Cancel your trial?" }),
    ).getByRole("button", { name: "Cancel trial" }),
  );
  await screen.findByText(/Cancellation confirmed/);
  expect(cancel).toHaveBeenCalledOnce();
  expect(screen.getByText("Your trial has ended")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Cancel trial" })).toBeNull();
});

test("does not surface an eligible offer after its trial has converted", async () => {
  mockStatus({ ...eligible(), converted: true });
  render(<SubscriptionStep />);
  await waitFor(() =>
    expect(posthog.capture).toHaveBeenCalledWith(
      "trial_offer_viewed",
      expect.any(Object),
    ),
  );
  expect(screen.queryByRole("button", { name: /start.*trial/i })).toBeNull();
  expect(screen.queryByRole("region", { name: "AutoGPT trial" })).toBeNull();
  expect(
    screen.getByRole("button", { name: "Continue to checkout" }),
  ).toBeDefined();
});

test.each(["BASIC", "BUSINESS"] as const)(
  "preserves the existing fallback for an eligible %s offer",
  async (tier) => {
    mockStatus(eligible({ ...trialOffer, tier }));
    render(<SubscriptionStep />);
    const fallback = within(
      await screen.findByRole("region", { name: "AutoGPT trial" }),
    );
    expect(
      fallback.getByRole("button", { name: "Start 7-day trial" }),
    ).toBeDefined();
    // Only Pro and Max trials get the trial welcome; this one sits beside the
    // paid checkout.
    expect(
      screen.queryByRole("button", { name: "Start my free trial" }),
    ).toBeNull();
    expect(
      screen.getByRole("button", { name: "Continue to checkout" }),
    ).toBeDefined();
  },
);

test("saves the personalized onboarding draft before starting trial checkout", async () => {
  mockStatus();
  let releaseSave: (() => void) | undefined;
  const flush = vi.fn(
    () =>
      new Promise<void>((resolve) => {
        releaseSave = resolve;
      }),
  );
  useOnboardingWizardStore.setState({ flushProgress: flush });
  const checkout = vi.fn();
  server.use(
    getPostTrialsStartTrialCheckoutMockHandler200(() => {
      checkout();
      return { url: "https://checkout.stripe.com/trial" };
    }),
  );
  render(<SubscriptionStep />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Start my free trial" }),
  );
  expect(flush).toHaveBeenCalledOnce();
  expect(checkout).not.toHaveBeenCalled();
  releaseSave?.();
  await waitFor(() =>
    expect(checkoutLocation.assign).toHaveBeenCalledWith(
      "https://checkout.stripe.com/trial",
    ),
  );
});

test("keeps the trial checkout available for retry if saving the draft fails", async () => {
  mockStatus();
  useOnboardingWizardStore.setState({
    flushProgress: vi.fn().mockRejectedValue(new Error("offline")),
  });
  render(<SubscriptionStep />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Start my free trial" }),
  );
  expect(
    (await screen.findByRole("alert")).textContent?.includes(
      "couldn't save your progress",
    ),
  ).toBe(true);
  expect(checkoutLocation.assign).not.toHaveBeenCalled();
  expect(
    screen
      .getByRole("button", { name: "Start my free trial" })
      .hasAttribute("disabled"),
  ).toBe(false);
});

test("also flushes the draft from the standalone fallback trial card", async () => {
  mockStatus(eligible({ ...trialOffer, tier: "BASIC" }));
  const flush = vi.fn().mockRejectedValue(new Error("offline"));
  useOnboardingWizardStore.setState({ flushProgress: flush });
  render(<SubscriptionStep />);
  const card = within(
    await screen.findByRole("region", { name: "AutoGPT trial" }),
  );
  fireEvent.click(
    await card.findByRole("button", { name: "Start 7-day trial" }),
  );
  expect(flush).toHaveBeenCalledOnce();
  expect(await card.findByRole("alert")).toBeDefined();
  expect(checkoutLocation.assign).not.toHaveBeenCalled();
});

test.each(["failed", "left with Back"])(
  "a %s paid checkout is cleared and then locked while the fallback trial redirects",
  async (outcome) => {
    mockStatus(eligible({ ...trialOffer, tier: "BASIC" }));
    server.use(
      http.post("*/api/credits/subscription", () =>
        outcome === "failed"
          ? HttpResponse.json({ detail: "Please try again" }, { status: 503 })
          : HttpResponse.json({ url: "https://checkout.stripe.com/paid" }),
      ),
      getPostTrialsStartTrialCheckoutMockHandler200({
        url: "https://checkout.stripe.com/trial",
      }),
    );
    render(<SubscriptionStep />);
    const fallback = within(
      await screen.findByRole("region", { name: "AutoGPT trial" }),
    );
    const start = await fallback.findByRole("button", {
      name: "Start 7-day trial",
    });
    const pro = screen.getByRole("button", { name: "Continue to checkout" });

    fireEvent.click(pro);
    if (outcome === "failed") {
      await waitFor(() => expect(pro.hasAttribute("disabled")).toBe(false));
      expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
    } else {
      await waitFor(() =>
        expect(checkoutLocation.href).toBe("https://checkout.stripe.com/paid"),
      );
      expect(pro.querySelector(".animate-spin")).not.toBeNull();
      restoreFromBackForwardCache();
      expect(pro.hasAttribute("disabled")).toBe(false);
    }
    expect(pro.querySelector(".animate-spin")).toBeNull();

    fireEvent.click(start);
    await waitFor(() =>
      expect(checkoutLocation.assign).toHaveBeenCalledWith(
        "https://checkout.stripe.com/trial",
      ),
    );
    expect(pro.hasAttribute("disabled")).toBe(true);
    expect(pro.querySelector(".animate-spin")).toBeNull();
  },
);
