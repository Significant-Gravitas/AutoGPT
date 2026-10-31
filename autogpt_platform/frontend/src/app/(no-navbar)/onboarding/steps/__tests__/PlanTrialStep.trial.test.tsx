import { getGetBrainDumpPlanMockHandler200 } from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import type { PlanResponse } from "@/app/api/__generated__/models/planResponse";
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
import {
  PAYWALL_LAST_STEPS,
  type Step,
  useOnboardingWizardStore,
} from "../../store";
import { PlanTrialStep } from "../SubscriptionStep/PlanTrialStep";
import { translateText } from "@/tests/integrations/translation";

// One capture for both clients: the trial offer reports through
// @posthog/react, the paywall and the plan through posthog-js.
const posthog = vi.hoisted(() => ({
  __loaded: true,
  is_capturing: () => true,
  capture: vi.fn(),
}));
vi.mock("@posthog/react", () => ({
  usePostHog: () => posthog,
}));
vi.mock("posthog-js", () => ({ default: posthog }));

const readyPlan: PlanResponse = {
  ready: true,
  source: "fallback",
  skipped_dump: false,
  heard: null,
  team: [{ template_id: "maria", name: "Maria", role: "Assistant" }],
  tasks: [
    {
      id: "task-1",
      title: "Draft this week's update",
      prompt: "Draft this week's update",
      expert_template_id: "maria",
    },
  ],
};

const originalLocation = window.location;
let checkoutLocation = {
  origin: "http://localhost",
  href: "http://localhost/",
  assign: vi.fn(),
};

beforeEach(() => {
  setTrialUser();
  posthog.capture.mockClear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().setSteps(PAYWALL_LAST_STEPS);
  useOnboardingWizardStore
    .getState()
    .goToStep(PAYWALL_LAST_STEPS.subscription as Step);
  vi.spyOn(environment, "isLocal").mockReturnValue(false);
  server.use(getGetBrainDumpPlanMockHandler200(readyPlan));
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

function eventsNamed(name: string) {
  return posthog.capture.mock.calls.filter(([event]) => event === name);
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
    render(<PlanTrialStep hasPaywall />);
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
    render(<PlanTrialStep hasPaywall />);
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
    render(<PlanTrialStep hasPaywall />);
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

test("reports the trial offer as viewed, and the step as one paywall view", async () => {
  mockStatus();
  render(<PlanTrialStep hasPaywall />);
  await screen.findByRole("button", { name: "Start my free trial" });
  await waitFor(() =>
    expect(eventsNamed("trial_offer_viewed")).toEqual([
      [
        "trial_offer_viewed",
        {
          trial_offer_version: trialOffer.version,
          subscription_tier: "PRO",
          trial_duration_days: 7,
          surface: "onboarding",
        },
      ],
    ]),
  );
  expect(eventsNamed("paywall_viewed")).toEqual([
    ["paywall_viewed", { surface: "onboarding" }],
  ]);
});

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
  render(<PlanTrialStep hasPaywall />);
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
  render(<PlanTrialStep hasPaywall />);
  expect(
    await screen.findByRole("heading", {
      level: 1,
      name: "Your first 14 days are on us",
    }),
  ).toBeDefined();
  expect(screen.getByText("14-day free trial")).toBeDefined();
  const timeline = within(
    screen.getByRole("list", { name: "How your trial works" }),
  );
  expect(timeline.getByText("Your first payment: ¥1,234")).toBeDefined();
  expect(
    timeline.getByText(/^Then ¥1,234 a year, plus any applicable tax\./),
  ).toBeDefined();
  expect(
    screen.getByText("Stripe will ask for a card. Nothing is charged today."),
  ).toBeDefined();
});

test("shows the trial timeline beside the plan: today, the reminder, then the first payment", async () => {
  mockStatus();
  render(<PlanTrialStep hasPaywall />);
  expect(
    await screen.findByRole("region", { name: "Your first 7 days are on us" }),
  ).toBeDefined();
  expect(screen.getByText("7-day free trial")).toBeDefined();
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
  // The plan's team joins when the trial starts, not on a subscription.
  expect(
    await screen.findByText(
      "Maria joins your team when your trial starts. Nothing runs until you pick a task.",
    ),
  ).toBeDefined();
});

test("the trial starts while Otto is still writing the plan", async () => {
  mockStatus();
  const planRequested = vi.fn();
  server.use(
    getGetBrainDumpPlanMockHandler200(() => {
      planRequested();
      return { ...readyPlan, ready: false, team: [], tasks: [] };
    }),
    getPostTrialsStartTrialCheckoutMockHandler200({
      url: "https://checkout.stripe.com/trial",
    }),
  );
  render(<PlanTrialStep hasPaywall />);
  const start = await screen.findByRole("button", {
    name: "Start my free trial",
  });
  await waitFor(() => expect(planRequested).toHaveBeenCalled());
  expect(screen.getByText("Otto is writing your plan")).toBeDefined();
  expect(start.hasAttribute("disabled")).toBe(false);

  fireEvent.click(start);

  await waitFor(() =>
    expect(checkoutLocation.assign).toHaveBeenCalledWith(
      "https://checkout.stripe.com/trial",
    ),
  );
  // Still pending: the trial never waited for the plan.
  expect(screen.getByText("Otto is writing your plan")).toBeDefined();
  expect(eventsNamed("plan_viewed")).toEqual([]);
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
  render(<PlanTrialStep hasPaywall />);
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
  render(<PlanTrialStep hasPaywall />);
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
  render(<PlanTrialStep hasPaywall />);
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
  "shows an eligible %s offer as the trial notice beside the Pro checkout",
  async (tier) => {
    mockStatus(eligible({ ...trialOffer, tier }));
    render(<PlanTrialStep hasPaywall />);
    const fallback = within(
      await screen.findByRole("region", { name: "AutoGPT trial" }),
    );
    expect(
      fallback.getByRole("button", { name: "Start 7-day trial" }),
    ).toBeDefined();
    // Only Pro and Max trials get the trial panel; this one sits in the paid
    // panel, above its checkout.
    expect(
      screen.queryByRole("button", { name: "Start my free trial" }),
    ).toBeNull();
    expect(
      screen.getByRole("heading", { level: 1, name: "Start with Pro" }),
    ).toBeDefined();
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
  render(<PlanTrialStep hasPaywall />);
  const start = await screen.findByRole("button", {
    name: "Start my free trial",
  });
  fireEvent.click(start);
  expect(flush).toHaveBeenCalledOnce();
  expect(checkout).not.toHaveBeenCalled();
  // Busy while the save runs, so a second click can't start another one.
  await waitFor(() => expect(start.hasAttribute("disabled")).toBe(true));
  releaseSave?.();
  await waitFor(() =>
    expect(checkoutLocation.assign).toHaveBeenCalledWith(
      "https://checkout.stripe.com/trial",
    ),
  );
  expect(checkout).toHaveBeenCalledOnce();
  expect(flush).toHaveBeenCalledOnce();
});

test("keeps the trial checkout available for retry if saving the draft fails", async () => {
  mockStatus();
  useOnboardingWizardStore.setState({
    flushProgress: vi.fn().mockRejectedValue(new Error("offline")),
  });
  render(<PlanTrialStep hasPaywall />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Start my free trial" }),
  );
  expect((await screen.findByRole("alert")).textContent).toBe(
    "We couldn't save your progress. Please try again before continuing to checkout.",
  );
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
  render(<PlanTrialStep hasPaywall />);
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
    render(<PlanTrialStep hasPaywall />);
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
