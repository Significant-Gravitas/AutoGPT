import { http, HttpResponse } from "msw";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { environment } from "@/services/environment";
import { useOnboardingWizardStore } from "../../store";
import {
  getSubscriptionPricingExperimentConfig,
  getSubscriptionPricingExperimentPlans,
} from "../SubscriptionStep/helpers";
import { SubscriptionStep } from "../SubscriptionStep/SubscriptionStep";

const postHog = vi.hoisted(() => ({
  variant: undefined as string | boolean | undefined,
}));

vi.mock("@posthog/react", () => ({
  useFeatureFlagVariantKey: () => postHog.variant,
}));

vi.mock("@/components/atoms/FadeIn/FadeIn", () => ({
  FadeIn: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
}));

vi.mock("@/components/atoms/AutoGPTLogo/AutoGPTLogo", () => ({
  AutoGPTLogo: () => <span>AutoGPTLogo</span>,
}));

afterEach(cleanup);

beforeEach(() => {
  postHog.variant = undefined;
  useOnboardingWizardStore.getState().reset();
  // The paywall is the first step.
  useOnboardingWizardStore.getState().goToStep(1);
  // Default tests to cloud mode so they exercise the Stripe Checkout path.
  // The local-bypass test below opts back into LOCAL.
  vi.spyOn(environment, "isLocal").mockReturnValue(false);
});

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
      buttonVariant: "secondary",
    });
    expect(max).toMatchObject({
      highlighted: false,
      badge: null,
      buttonVariant: "secondary",
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
      buttonVariant: "secondary",
    });
  });
});

describe("SubscriptionStep", () => {
  test("renders the three plan cards by display name", () => {
    render(<SubscriptionStep />);
    expect(screen.getByRole("heading", { name: /^Pro$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Max$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Team$/ })).toBeDefined();
  });

  test("defaults to monthly billing with the full monthly price and a 'billed monthly' caption", () => {
    render(<SubscriptionStep />);
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("monthly");
    expect(screen.getByLabelText("$50.00")).toBeDefined();
    expect(screen.getByLabelText("$320.00")).toBeDefined();
    expect(screen.getAllByText("billed monthly").length).toBe(2);
    expect(screen.queryByText(/Charged today/i)).toBeNull();
  });

  test("highlights no plan by default — no 'Best value' badge", () => {
    render(<SubscriptionStep />);
    expect(screen.queryByText(/Best value/i)).toBeNull();
  });

  test("PostHog monthly Pro variant starts on monthly billing", async () => {
    postHog.variant = "monthly-pro";

    render(<SubscriptionStep />);

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "monthly",
      );
    });
    expect(screen.getByLabelText("$50.00")).toBeDefined();
    expect(screen.getAllByText("billed monthly").length).toBe(2);
  });

  test("PostHog billing default does not override a user-selected cycle", async () => {
    postHog.variant = "monthly-pro";
    useOnboardingWizardStore.getState().setSelectedBilling("yearly");

    render(<SubscriptionStep />);

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedBilling).toBe(
        "yearly",
      );
    });
    expect(screen.getByLabelText("$42.50")).toBeDefined();
  });

  test("switching to monthly shows the full monthly price and a 'billed monthly' caption", () => {
    render(<SubscriptionStep />);
    fireEvent.click(screen.getByRole("button", { name: /Monthly billing/i }));
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("monthly");
    expect(screen.getByLabelText("$50.00")).toBeDefined();
    expect(screen.getByLabelText("$320.00")).toBeDefined();
    expect(screen.getAllByText("billed monthly").length).toBe(2);
    expect(screen.queryByText(/Charged today/i)).toBeNull();
  });

  test("selecting Pro persists selectedPlan and redirects to Stripe Checkout (Welcome on success, paywall on cancel)", async () => {
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
    fireEvent.click(screen.getByRole("button", { name: /Get Pro/i }));

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });

    expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");
    expect(capturedTierBody!.tier).toBe("PRO");
    // Success returns to Welcome (step 2) to begin onboarding; cancel returns
    // to the paywall (step 1).
    expect(capturedTierBody!.success_url).toContain(
      "/onboarding?step=2&subscription=success",
    );
    expect(capturedTierBody!.cancel_url).toContain(
      "/onboarding?step=1&subscription=cancelled",
    );
    // Paywall-first: no profile data exists yet, so nothing is POSTed here —
    // the Preparing step submits the profile at the end of onboarding.
    expect(profileCalled).toBe(false);
  });

  test("switching to yearly + selecting Pro forwards billing_cycle=yearly", async () => {
    let capturedTierBody: {
      tier?: string;
      billing_cycle?: string;
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
    fireEvent.click(screen.getByRole("button", { name: /Yearly billing/i }));
    fireEvent.click(screen.getByRole("button", { name: /Get Pro/i }));

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    expect(capturedTierBody!.tier).toBe("PRO");
    expect(capturedTierBody!.billing_cycle).toBe("yearly");
  });

  test("switching to monthly + selecting Pro forwards billing_cycle=monthly", async () => {
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
    fireEvent.click(screen.getByRole("button", { name: /Monthly billing/i }));
    fireEvent.click(screen.getByRole("button", { name: /Get Pro/i }));

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    expect(capturedTierBody!.billing_cycle).toBe("monthly");
  });

  test("selecting Max uses the MAX tier in the Stripe Checkout request", async () => {
    let capturedTierBody: { tier?: string } | null = null;

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
    fireEvent.click(screen.getByRole("button", { name: /Upgrade to Max/i }));

    await waitFor(() => {
      expect(capturedTierBody).not.toBeNull();
    });
    expect(capturedTierBody!.tier).toBe("MAX");
    expect(useOnboardingWizardStore.getState().selectedPlan).toBe("MAX");
  });

  test("selecting Team opens the intake form and does not advance", () => {
    const openSpy = vi.spyOn(window, "open").mockImplementation(() => null);
    try {
      render(<SubscriptionStep />);
      fireEvent.click(screen.getByRole("button", { name: /Contact sales/i }));
      expect(openSpy).toHaveBeenCalledWith(
        expect.stringContaining("tally.so"),
        "_blank",
        "noopener,noreferrer",
      );
      const state = useOnboardingWizardStore.getState();
      expect(state.selectedPlan).toBeNull();
      expect(state.currentStep).toBe(1);
    } finally {
      openSpy.mockRestore();
    }
  });

  test("local dev: clicking Pro skips Stripe and advances to the next step", async () => {
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
    fireEvent.click(screen.getByRole("button", { name: /Get Pro/i }));

    await waitFor(() => {
      expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");
    });
    // Local short-circuit: no Stripe Checkout, no profile POST (the Preparing
    // step handles submission via useOnboardingPage). Advances from the
    // paywall (step 1) to Welcome (step 2).
    expect(stripeCalled).toBe(false);
    expect(profileCalledSync).toBe(false);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(2);
  });

  test("clicking a plan keeps the request in flight: clicked card spins, others lock", async () => {
    useOnboardingWizardStore.getState().setName("Ada");
    useOnboardingWizardStore.getState().setRole("Engineer");

    let resolveTier: (value: unknown) => void = () => undefined;
    server.use(
      http.post("*/api/onboarding/profile", () =>
        HttpResponse.json({}, { status: 200 }),
      ),
      http.post(
        "*/api/credits/subscription",
        () =>
          new Promise((resolve) => {
            resolveTier = () => resolve(HttpResponse.json({ url: null }));
          }),
      ),
    );

    render(<SubscriptionStep />);
    const proButton = screen.getByRole("button", { name: /Get Pro/i });
    const maxButton = screen.getByRole("button", { name: /Upgrade to Max/i });
    const teamButton = screen.getByRole("button", { name: /Contact sales/i });
    fireEvent.click(proButton);

    await waitFor(() => {
      expect(maxButton.hasAttribute("disabled")).toBe(true);
      expect(teamButton.hasAttribute("disabled")).toBe(true);
    });
    expect(useOnboardingWizardStore.getState().selectedPlan).toBe("PRO");

    resolveTier(null);
  });
});
