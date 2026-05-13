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
import { SubscriptionStep } from "../SubscriptionStep/SubscriptionStep";

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
  useOnboardingWizardStore.getState().reset();
  useOnboardingWizardStore.getState().goToStep(4);
  // Default tests to cloud mode so they exercise the Stripe Checkout path.
  // The local-bypass test below opts back into LOCAL.
  vi.spyOn(environment, "isLocal").mockReturnValue(false);
});

describe("SubscriptionStep", () => {
  test("renders the three plan cards by display name", () => {
    render(<SubscriptionStep />);
    expect(screen.getByRole("heading", { name: /^Pro$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Max$/ })).toBeDefined();
    expect(screen.getByRole("heading", { name: /^Team$/ })).toBeDefined();
  });

  test("defaults to yearly billing with the monthly-equivalent price and the annual charge", () => {
    render(<SubscriptionStep />);
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("yearly");
    expect(screen.getByText("$42.50")).toBeDefined();
    expect(screen.getByText("$272.00")).toBeDefined();
    expect(screen.getByText("Charged today: $510.00")).toBeDefined();
    expect(screen.getByText("Charged today: $3,264.00")).toBeDefined();
    expect(screen.getAllByText(/Save 15%/).length).toBeGreaterThan(0);
  });

  test("switching to monthly shows the full monthly price and matching charged-today", () => {
    render(<SubscriptionStep />);
    fireEvent.click(screen.getByRole("button", { name: /Monthly billing/i }));
    expect(useOnboardingWizardStore.getState().selectedBilling).toBe("monthly");
    expect(screen.getByText("$50.00")).toBeDefined();
    expect(screen.getByText("$320.00")).toBeDefined();
    expect(screen.getByText("Charged today: $50.00")).toBeDefined();
    expect(screen.getByText("Charged today: $320.00")).toBeDefined();
  });

  test("selecting Pro persists selectedPlan, submits the profile, and redirects to Stripe Checkout", async () => {
    useOnboardingWizardStore.getState().setName("Ada Lovelace");
    useOnboardingWizardStore.getState().setRole("Engineer");
    useOnboardingWizardStore.getState().togglePainPoint("Repetitive work");

    let capturedTierBody: {
      tier?: string;
      success_url?: string;
      cancel_url?: string;
    } | null = null;
    let capturedProfileBody: {
      user_name?: string;
      user_role?: string;
      pain_points?: string[];
    } | null = null;

    server.use(
      http.post("*/api/onboarding/profile", async ({ request }) => {
        capturedProfileBody =
          (await request.json()) as typeof capturedProfileBody;
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
    expect(capturedTierBody!.success_url).toContain(
      "/onboarding?step=5&subscription=success",
    );
    expect(capturedTierBody!.cancel_url).toContain(
      "/onboarding?step=4&subscription=cancelled",
    );
    expect(capturedProfileBody).not.toBeNull();
    expect(capturedProfileBody!.user_name).toBe("Ada Lovelace");
    expect(capturedProfileBody!.user_role).toBe("Engineer");
    expect(capturedProfileBody!.pain_points).toEqual(["Repetitive work"]);
  });

  test("default yearly + selecting Pro forwards billing_cycle=yearly", async () => {
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
      expect(state.currentStep).toBe(4);
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
    // Local short-circuit: no Stripe Checkout, no pre-redirect profile POST
    // (the Preparing step handles submission via useOnboardingPage).
    expect(stripeCalled).toBe(false);
    expect(profileCalledSync).toBe(false);
    expect(useOnboardingWizardStore.getState().currentStep).toBe(5);
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
