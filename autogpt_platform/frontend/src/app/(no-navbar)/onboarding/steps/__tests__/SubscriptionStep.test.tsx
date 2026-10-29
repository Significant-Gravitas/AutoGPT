import { http, HttpResponse } from "msw";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import { restoreFromBackForwardCache } from "@/tests/integrations/back-forward-cache";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import { useOnboardingWizardStore } from "../../store";
import {
  getReportedPricingVariant,
  getSubscriptionPricingExperimentConfig,
  getSubscriptionPricingExperimentPlans,
} from "../SubscriptionStep/helpers";
import { SubscriptionStep } from "../SubscriptionStep/SubscriptionStep";
import { getGetExperimentsGetExperimentAssignmentMockHandler } from "@/app/api/__generated__/endpoints/experiments/experiments.msw";
import { getGetTrialsGetTrialStatusMockHandler200 } from "@/app/api/__generated__/endpoints/trials/trials.msw";
import {
  getGetBrainDumpRecommendedExpertsMockHandler200,
  getGetBrainDumpRecommendedExpertsResponseMock200,
} from "@/app/api/__generated__/endpoints/brain-dump/brain-dump.msw";
import { Experiment } from "@/app/api/__generated__/models/experiment";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";

vi.mock("@posthog/react", () => ({
  usePostHog: () => undefined,
}));

const posthogJS = vi.hoisted(() => ({
  __loaded: true,
  is_capturing: () => true,
  capture: vi.fn(),
}));
vi.mock("posthog-js", () => ({ default: posthogJS }));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("@/components/atoms/AutoGPTLogo/AutoGPTLogo", () => ({
  AutoGPTLogo: () => <span>AutoGPTLogo</span>,
}));

afterEach(() => {
  // Not at the end of each test body: an assertion that throws above would
  // leak the shim, the location stub and NEXT_PUBLIC_GOOGLE_ADS_ID into every
  // later test here.
  restoreLocation();
  removeGtagShim();
  vi.unstubAllEnvs();
  useAuthStore.setState({ user: null, hasLoadedUser: false });
});

beforeEach(() => {
  posthogJS.capture.mockClear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  // The paywall is the last interactive step (step 3), before Preparing.
  useOnboardingWizardStore.getState().goToStep(3);
  // Default tests to cloud mode so they exercise the Stripe Checkout path.
  // The local-bypass test below opts back into LOCAL.
  vi.spyOn(environment, "isLocal").mockReturnValue(false);
});

const originalLocation = window.location;

// The success path hands off to Stripe via window.location.href; jsdom tears
// the environment down on a real navigation, so swap in a plain object.
function stubLocation() {
  const location = { origin: "http://localhost", href: "http://localhost/" };
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: location,
  });
  return location;
}

function restoreLocation() {
  Object.defineProperty(window, "location", {
    configurable: true,
    writable: true,
    value: originalLocation,
  });
}

describe("subscription pricing experiment helpers", () => {
  test("defaults to monthly billing with no highlighted plan (matches the paywall)", () => {
    expect(getSubscriptionPricingExperimentConfig(undefined)).toMatchObject({
      billing: "monthly",
      highlightedPlan: null,
      variant: "control",
    });
  });

  test("highlights no plan when highlightedPlan is null", () => {
    const plans = getSubscriptionPricingExperimentPlans(null);
    const pro = plans.find((plan) => plan.key === "PRO");
    const max = plans.find((plan) => plan.key === "MAX");

    expect(pro).toMatchObject({
      highlighted: false,
      badge: null,
      buttonVariant: "primary",
    });
    expect(max).toMatchObject({
      highlighted: false,
      badge: null,
      buttonVariant: "primary",
    });
  });

  test("maps PostHog variants to billing and highlighted plan config", () => {
    expect(getSubscriptionPricingExperimentConfig("monthly-pro")).toMatchObject(
      {
        billing: "monthly",
        highlightedPlan: "PRO",
        variant: "monthly-pro",
      },
    );
    expect(getSubscriptionPricingExperimentConfig("yearly-max")).toMatchObject({
      billing: "yearly",
      highlightedPlan: "MAX",
      variant: "yearly-max",
    });
  });

  test("moves the highlighted styling from Max to Pro", () => {
    const plans = getSubscriptionPricingExperimentPlans("PRO");
    const pro = plans.find((plan) => plan.key === "PRO");
    const max = plans.find((plan) => plan.key === "MAX");

    expect(pro).toMatchObject({
      highlighted: true,
      badge: "Best value",
      buttonVariant: "primary",
    });
    expect(max).toMatchObject({
      highlighted: false,
      badge: null,
      buttonVariant: "primary",
    });
  });

  test("reports a pricing arm only once PostHog has assigned a real one", () => {
    expect(getReportedPricingVariant("yearly-max", true)).toBe("yearly-max");
    expect(getReportedPricingVariant("control", true)).toBe("control");
    // Still loading flags, not enrolled, or an arm this page does not know.
    expect(getReportedPricingVariant("yearly-max", false)).toBeUndefined();
    expect(getReportedPricingVariant(null, true)).toBeUndefined();
    expect(getReportedPricingVariant("annual-team", true)).toBeUndefined();
  });
});

describe("SubscriptionStep", () => {
  test("offers one Pro checkout, with no Max or Team plan and no billing toggle", () => {
    render(<SubscriptionStep />);
    expect(
      screen.getByRole("heading", { name: "Put your setup to work" }),
    ).toBeDefined();
    expect(checkoutButton()).toBeDefined();
    expect(screen.queryByText(/\bMax\b/)).toBeNull();
    expect(screen.queryByText(/\bTeam\b/)).toBeNull();
    expect(screen.queryByRole("button", { name: /billing/i })).toBeNull();
  });

  test("defaults to monthly billing and shows the monthly Pro price plus tax", () => {
    render(<SubscriptionStep />);
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("monthly");
    expect(screen.getByText("You pay $50")).toBeDefined();
    expect(screen.getByText(/Plus any applicable tax/)).toBeDefined();
    expect(screen.getByText("$50 a month until you cancel.")).toBeDefined();
  });

  test("an assigned monthly Pro arm starts on monthly billing", async () => {
    assignArm("monthly-pro");

    render(<SubscriptionStep />);

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "monthly",
      );
    });
    expect(await screen.findByText("You pay $50")).toBeDefined();
  });

  test("an assigned arm's billing default does not override a user-selected cycle", async () => {
    assignArm("monthly-pro");
    useOnboardingWizardStore.getState().setSelectedBilling("yearly");

    render(<SubscriptionStep />);

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "yearly",
      );
    });
    expect(await screen.findByText("You pay $510")).toBeDefined();
    expect(screen.getByText("$510 a year until you cancel.")).toBeDefined();
  });

  test("names the hired Experts in the ready line", async () => {
    useOnboardingWizardStore.setState({ hiredTemplateIds: ["maria", "mina"] });
    server.use(
      getGetBrainDumpRecommendedExpertsMockHandler200(
        getGetBrainDumpRecommendedExpertsResponseMock200({
          team: {
            diagnosis: "Busy founder",
            experts: [
              recommendedExpert("maria", "Maria"),
              recommendedExpert("mina", "Mina"),
              recommendedExpert("not-hired", "Nadia"),
            ],
            source: "llm",
          },
        }),
      ),
    );

    render(<SubscriptionStep />);

    expect(
      // The list follows the browser's locale, so the serial comma varies.
      await screen.findByText(/^Otto, Maria,? and Mina are ready to start$/),
    ).toBeDefined();
    expect(screen.queryByText(/Nadia/)).toBeNull();
  });

  test("selecting Pro persists selectedPlan and returns to Preparing on success, paywall on cancel", async () => {
    let capturedTierBody: {
      tier?: string;
      success_url?: string;
      cancel_url?: string;
    } | null = null;
    let profileCalled = false;

    server.use(
      http.post("*/api/onboarding/profile", () => {
        profileCalled = true;
        return HttpResponse.json({}, { status: 200 });
      }),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedTierBody = (await request.json()) as typeof capturedTierBody;
        // Return null url so the hook doesn't try to navigate window.location
        // (would tear down the test environment).
        return HttpResponse.json({ url: null });
      }),
    );

    render(<SubscriptionStep />);
    fireEvent.click(checkoutButton());

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });

    expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");
    expect(capturedTierBody!.tier).toBe("PRO");
    // Checkout returns to semantic steps so optional layout changes are safe.
    expect(capturedTierBody!.success_url).toContain(
      "/onboarding?step=preparing&subscription=success",
    );
    expect(capturedTierBody!.cancel_url).toContain(
      "/onboarding?step=subscription&subscription=cancelled",
    );
    // Stripe fills {CHECKOUT_SESSION_ID}; plan and cycle let the return page
    // report the subscription value to Google Ads.
    expect(capturedTierBody!.success_url).toContain(
      "&session_id={CHECKOUT_SESSION_ID}&plan=PRO&cycle=monthly",
    );
    // Answers have already been collected; Preparing submits the final profile.
    expect(profileCalled).toBe(false);
  });

  test("reports begin_checkout with the plan price to Google Ads before redirecting", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
    const gtagCalls = installGtagShim();
    const location = stubLocation();
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json({ url: "https://checkout.stripe.com/pay/cs_test" }),
      ),
    );

    render(<SubscriptionStep />);
    fireEvent.click(checkoutButton());

    await waitFor(() => {
      expect(gtagCalls).toContainEqual([
        "event",
        "conversion",
        {
          send_to: "AW-123/BC",
          value: 50,
          currency: "USD",
          event_callback: expect.any(Function),
        },
      ]);
    });
    await waitFor(() =>
      expect(location.href).toBe("https://checkout.stripe.com/pay/cs_test"),
    );
  });

  test("stays busy while Stripe loads, and is usable again after Back restores the page", async () => {
    const location = stubLocation();
    let posts = 0;
    server.use(
      http.post("*/api/credits/subscription", () => {
        posts += 1;
        return HttpResponse.json({
          url: "https://checkout.stripe.com/pay/cs_test",
        });
      }),
    );

    render(<SubscriptionStep />);
    const pro = checkoutButton();
    fireEvent.click(pro);
    await waitFor(() =>
      expect(location.href).toBe("https://checkout.stripe.com/pay/cs_test"),
    );
    expect(pro.hasAttribute("disabled")).toBe(true);
    fireEvent.click(pro);
    expect(posts).toBe(1);

    restoreFromBackForwardCache();

    expect(pro.hasAttribute("disabled")).toBe(false);
    fireEvent.click(pro);
    await waitFor(() => expect(posts).toBe(2));
  });

  test.each(["https://checkout.stripe.com/pay/old_account", null])(
    "ignores an old account's checkout response with URL %s",
    async (url) => {
      const location = stubLocation();
      let resolveResponse: (() => void) | undefined;
      const pending = new Promise<void>((resolve) => {
        resolveResponse = resolve;
      });
      const received = vi.fn();
      server.use(
        http.post("*/api/credits/subscription", async () => {
          received();
          await pending;
          return HttpResponse.json({ url });
        }),
      );
      useOnboardingWizardStore.setState({ userID: "previous-user" });
      render(<SubscriptionStep />);
      fireEvent.click(checkoutButton());
      await waitFor(() => expect(received).toHaveBeenCalledOnce());
      act(() => {
        useOnboardingWizardStore.setState({
          userID: "next-user",
          currentStep: 1,
        });
        resolveResponse?.();
      });
      await waitFor(() =>
        expect(checkoutButton().hasAttribute("disabled")).toBe(false),
      );
      expect(location.href).toBe("http://localhost/");
      expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
    },
  );

  test("reports no begin_checkout when Stripe returns no Checkout URL", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS", "begin_checkout=BC");
    const gtagCalls = installGtagShim();
    // The backend modified the subscription in place — no Checkout ever
    // started, so counting it would inflate the funnel.
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json({ url: null }),
      ),
    );

    render(<SubscriptionStep />);
    fireEvent.click(checkoutButton());

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
    });
    expect(gtagCalls.filter((call) => call[1] === "conversion")).toEqual([]);
  });

  test("a yearly pricing arm forwards billing_cycle=yearly to checkout", async () => {
    assignArm("yearly-pro");
    let capturedTierBody: {
      tier?: string;
      billing_cycle?: string;
      success_url?: string;
    } | null = null;

    server.use(
      http.post("*/api/onboarding/profile", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedTierBody = (await request.json()) as typeof capturedTierBody;
        return HttpResponse.json({ url: null });
      }),
    );

    render(<SubscriptionStep />);
    expect(await screen.findByText("You pay $510")).toBeDefined();
    fireEvent.click(checkoutButton());

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    expect(capturedTierBody!.tier).toBe("PRO");
    expect(capturedTierBody!.billing_cycle).toBe("yearly");
    expect(capturedTierBody!.success_url).toContain("&plan=PRO&cycle=yearly");
  });

  test("a monthly pricing arm forwards billing_cycle=monthly to checkout", async () => {
    assignArm("monthly-pro");
    let capturedTierBody: { billing_cycle?: string } | null = null;

    server.use(
      http.post("*/api/onboarding/profile", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedTierBody = (await request.json()) as typeof capturedTierBody;
        return HttpResponse.json({ url: null });
      }),
    );

    render(<SubscriptionStep />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue to checkout" }),
    );

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    expect(capturedTierBody!.billing_cycle).toBe("monthly");
  });

  test("reports the paywall view and the picked plan with its pricing arm to PostHog", async () => {
    assignArm("yearly-max");
    let capturedTierBody: { surface?: string } | null = null;
    server.use(
      http.post("*/api/credits/subscription", async ({ request }) => {
        capturedTierBody = (await request.json()) as typeof capturedTierBody;
        return HttpResponse.json({ url: null });
      }),
    );

    render(<SubscriptionStep />);
    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "yearly",
      );
    });
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue to checkout" }),
    );

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    // The backend reads `surface` for its checkout_started event.
    expect(capturedTierBody!.surface).toBe("onboarding");
    expect(posthogJS.capture).toHaveBeenCalledWith("paywall_viewed", {
      surface: "onboarding",
    });
    expect(posthogJS.capture).toHaveBeenCalledWith("plan_selected", {
      subscription_tier: "PRO",
      billing_cycle: "yearly",
      surface: "onboarding",
      pricing_variant: "yearly-max",
    });
  });

  // A user outside the rollout sees the control layout but is not in the control arm.
  test("reports no pricing arm for a user the experiment did not enrol", async () => {
    assignArm(null);
    server.use(
      http.post("*/api/credits/subscription", () =>
        HttpResponse.json({ url: null }),
      ),
    );

    render(<SubscriptionStep />);
    fireEvent.click(
      await screen.findByRole("button", { name: "Continue to checkout" }),
    );

    await waitFor(() => {
      expect(posthogJS.capture).toHaveBeenCalledWith("plan_selected", {
        subscription_tier: "PRO",
        billing_cycle: "monthly",
        surface: "onboarding",
      });
    });
  });

  test("local dev: continuing skips Stripe and advances to the next step", async () => {
    vi.spyOn(environment, "isLocal").mockReturnValue(true);

    let stripeCalled = false;
    let profileCalledSync = false;
    server.use(
      http.post("*/api/credits/subscription", () => {
        stripeCalled = true;
        return HttpResponse.json({ url: null });
      }),
      http.post("*/api/onboarding/profile", () => {
        profileCalledSync = true;
        return HttpResponse.json({}, { status: 200 });
      }),
    );

    render(<SubscriptionStep />);
    fireEvent.click(checkoutButton());

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");
    });
    // Local short-circuit: no Stripe Checkout, no profile POST (the Preparing
    // step handles submission via useOnboardingPage). Advances from the
    // paywall (step 3) to Preparing (step 4).
    expect(stripeCalled).toBe(false);
    expect(profileCalledSync).toBe(false);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
  });

  test("keeps the checkout button locked while the request is in flight", async () => {
    useOnboardingWizardStore.getState().setRole("Engineer");

    let posts = 0;
    let resolveTier: (value: unknown) => void = () => undefined;
    server.use(
      http.post("*/api/onboarding/profile", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
      http.post("*/api/credits/subscription", () => {
        posts += 1;
        return new Promise((resolve) => {
          resolveTier = () => resolve(HttpResponse.json({ url: null }));
        });
      }),
    );

    render(<SubscriptionStep />);
    const button = checkoutButton();
    fireEvent.click(button);

    await waitFor(() => {
      expect(button.hasAttribute("disabled")).toBe(true);
    });
    fireEvent.click(button);
    await waitFor(() => expect(posts).toBe(1));
    expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");

    resolveTier(null);
    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().currentStep).toBe(4);
    });
    expect(posts).toBe(1);
  });
});

function checkoutButton() {
  return screen.getByRole("button", { name: "Continue to checkout" });
}

function recommendedExpert(templateID: string, name: string) {
  return {
    template_id: templateID,
    name,
    role: "Assistant",
    avatar_url: null,
    reason: "Fits your goals",
    workflow_names: [],
  };
}

// Signs in a user with no trial, whom the backend assigned *variant* (null: not enrolled).
function assignArm(variant: string | null) {
  useAuthStore.setState({
    user: { id: "user-1", email: "user-1@example.com", user_metadata: {} },
    isUserLoading: false,
    // Without it the store refetches the session, which this suite answers with nobody.
    hasLoadedUser: true,
  });
  server.use(
    getGetTrialsGetTrialStatusMockHandler200({
      eligible: false,
      active: false,
    }),
    getGetExperimentsGetExperimentAssignmentMockHandler(
      variant === null
        ? null
        : {
            experiment_key: Experiment.SUBSCRIPTION_PRICING,
            variant,
            source: "backend",
            assigned_at: new Date("2026-10-11T00:00:00Z"),
          },
    ),
  );
}
