import {
  getGetTrialsGetTrialStatusMockHandler200,
  getPostTrialsCancelTrialMockHandler200,
  getPostTrialsStartTrialCheckoutMockHandler200,
} from "@/app/api/__generated__/endpoints/trials/trials.msw";
import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
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

const posthog = vi.hoisted(() => ({ capture: vi.fn() }));
vi.mock("@posthog/react", () => ({
  useFeatureFlagVariantKey: () => undefined,
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

test.each([
  ["PRO", "Pro"],
  ["MAX", "Max"],
] as const)(
  "starts the integrated %s trial with the accepted token and onboarding destination",
  async (tier, name) => {
    mockStatus(eligible({ ...trialOffer, tier }));
    const requestBody = vi.fn();
    server.use(
      getPostTrialsStartTrialCheckoutMockHandler200(async ({ request }) => {
        requestBody(await request.json());
        return { url: "https://checkout.stripe.com/trial" };
      }),
    );
    render(<SubscriptionStep />);
    const card = within(
      await screen.findByRole("region", { name: `${name} plan` }),
    );
    const start = await card.findByRole("button", {
      name: "Start 7-day trial",
    });
    expect(start.getAttribute("data-fast-goal")).not.toBe("plan_cta_click");
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
    expect(posthog.capture).toHaveBeenCalledWith(
      "subscription_trial_checkout_started",
      { trial_offer_version: trialOffer.version, surface: "onboarding" },
    );
    expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
  },
);

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
  useOnboardingWizardStore.getState().setSelectedBilling("yearly");
  render(<SubscriptionStep />);
  const max = within(await screen.findByRole("region", { name: "Max plan" }));
  await max.findByRole("button", { name: "Start 14-day trial" });
  expect(max.getByText("¥1,234")).toBeDefined();
  expect(max.getByText("/ year")).toBeDefined();
  fireEvent.click(max.getByRole("button", { name: "Trial details" }));
  const details = (await screen.findByRole("dialog")).textContent ?? "";
  expect(details).toMatch(/14/);
  expect(details).toMatch(/¥1,234\s*\/\s*year/);
  expect(details).toMatch(/card required/i);
  expect(details).toMatch(/applicable tax/i);
  expect(details).toMatch(/cancel/i);
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
    await screen.findByRole("button", { name: "Start 7-day trial" }),
  );
  expect(await screen.findByRole("alert")).toBeDefined();
  fireEvent.click(
    await screen.findByRole("button", { name: "Start 10-day trial" }),
  );
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

test("retains paid checkout, Google Ads value, and DataFast metadata while a trial is eligible", async () => {
  mockStatus();
  vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
  vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
  const gtag = installGtagShim();
  const paid = vi.fn();
  server.use(
    http.post("*/api/credits/subscription", async ({ request }) => {
      paid(await request.json());
      return HttpResponse.json({ url: "https://checkout.stripe.com/paid" });
    }),
  );
  render(<SubscriptionStep />);
  await screen.findByRole("button", { name: "Start 7-day trial" });
  const pro = within(
    screen.getByRole("region", { name: "Pro plan" }),
  ).getByRole("button", { name: /^Get Pro/ });
  expect(pro.getAttribute("data-fast-goal")).toBe("plan_cta_click");
  expect(pro.getAttribute("data-fast-goal-plan")).toBe("pro");
  expect(pro.getAttribute("data-fast-goal-cycle")).toBe("monthly");
  expect(pro.getAttribute("data-fast-goal-surface")).toBe("onboarding_paywall");
  fireEvent.click(pro);
  await waitFor(() =>
    expect(checkoutLocation.href).toBe("https://checkout.stripe.com/paid"),
  );
  expect(paid).toHaveBeenCalledWith(
    expect.objectContaining({ tier: "PRO", billing_cycle: "monthly" }),
  );
  expect(gtag).toContainEqual([
    "event",
    "conversion",
    { send_to: "AW-123/BC", value: 50, currency: "USD" },
  ]);
  expect(checkoutLocation.assign).not.toHaveBeenCalled();
});

test("switches to the offered billing cycle without starting either checkout", async () => {
  mockStatus();
  useOnboardingWizardStore.getState().setSelectedBilling("yearly");
  render(<SubscriptionStep />);
  const alternative = await screen.findByRole("button", {
    name: "Try Pro for 7 days instead",
  });
  expect(
    screen.queryByRole("button", { name: "Start 7-day trial" }),
  ).toBeNull();
  const toggle = screen.getByRole("button", { name: /Monthly billing/i });
  expect(toggle.getAttribute("data-fast-goal")).toBe("paywall_billing_toggle");
  expect(toggle.getAttribute("data-fast-goal-cycle")).toBe("monthly");
  fireEvent.click(alternative);
  expect(useOnboardingWizardStore.getState().selectedBilling).toBe("monthly");
  await screen.findByRole("button", { name: "Start 7-day trial" });
  expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
  expect(checkoutLocation.assign).not.toHaveBeenCalled();
  expect(checkoutLocation.href).toBe("http://localhost/");
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
    screen.queryByRole("button", { name: "Start 7-day trial" }),
  ).toBeNull();
  expect(cancel).not.toHaveBeenCalled();
  fireEvent.click(await screen.findByRole("button", { name: "End trial now" }));
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
  expect(screen.getByRole("button", { name: "Get Pro" })).toBeDefined();
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
    expect(
      within(screen.getByRole("region", { name: "Pro plan" })).queryByRole(
        "button",
        { name: /start.*trial/i },
      ),
    ).toBeNull();
  },
);
