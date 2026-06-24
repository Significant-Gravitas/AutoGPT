import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";

vi.mock("../steps/WelcomeStep", () => ({
  WelcomeStep: () => <div data-testid="step-welcome" />,
}));
vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => <div data-testid="step-role" />,
}));
vi.mock("../steps/PainPointsStep", () => ({
  PainPointsStep: () => <div data-testid="step-painpoints" />,
}));
vi.mock("../steps/SubscriptionStep/SubscriptionStep", () => ({
  SubscriptionStep: () => <div data-testid="step-subscription" />,
}));
vi.mock("../steps/PreparingStep", () => ({
  PreparingStep: ({ onComplete: _onComplete }: { onComplete: () => void }) => (
    <div data-testid="step-preparing" />
  ),
}));

let currentSearchParams = new URLSearchParams();
const routerReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => currentSearchParams,
  usePathname: () => "/onboarding",
}));

let mockSupabaseState = { isLoggedIn: true, isUserLoading: false };
vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ ...mockSupabaseState, user: null }),
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: () =>
    Promise.resolve({ status: 200, data: { completedSteps: [] } }),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
  postV1SubmitOnboardingProfile: () => Promise.resolve({ status: 200 }),
}));

let mockSubscriptionTier: string = "NO_TIER";
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  useGetSubscriptionStatus: (opts: {
    query: { select: (res: { status: number; data: unknown }) => unknown };
  }) => ({
    data: opts.query.select({
      status: 200,
      data: { tier: mockSubscriptionTier },
    }),
    isLoading: false,
  }),
}));

vi.mock("@/app/api/helpers", () => ({
  resolveResponse: (p: Promise<{ data: unknown }>) => p.then((r) => r.data),
}));

let mockFlagValue = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT" },
  useGetFlag: () => mockFlagValue,
}));

vi.mock("launchdarkly-react-client-sdk", () => ({
  useLDClient: () => ({
    waitForInitialization: () => Promise.resolve(),
  }),
}));

const STEP_STORAGE_KEY = "autogpt:onboarding-highest-step";

beforeEach(() => {
  currentSearchParams = new URLSearchParams();
  mockFlagValue = false;
  mockSubscriptionTier = "NO_TIER";
  mockSupabaseState = { isLoggedIn: true, isUserLoading: false };
  routerReplace.mockClear();
  useOnboardingWizardStore.getState().reset();
  window.sessionStorage.removeItem(STEP_STORAGE_KEY);
});

afterEach(() => {
  cleanup();
});

describe("OnboardingPage — flag-gated SubscriptionStep", () => {
  it("renders Welcome at step 1 by default in flag-off mode", async () => {
    mockFlagValue = false;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=5 to step 1 when payments are gated off", async () => {
    mockFlagValue = false;
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("treats step 4 as Preparing when payments are gated off", async () => {
    mockFlagValue = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("renders SubscriptionStep first (step 1) when payments are enabled", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=1");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-welcome")).toBeNull();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("renders Welcome at step 2 (after the paywall) when payments are enabled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
    currentSearchParams = new URLSearchParams("step=2");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("treats step 5 as Preparing when payments are enabled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "5");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=5 to the user's highest reached step (no fast-forward)", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    // Highest reached is 3 (RoleStep), so manually editing the URL to step=5
    // should land the user back on step 3, not let them skip ahead.
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("resumes from the highest reached step when ?step= is omitted", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    // No step param — user navigated to /onboarding directly.
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("rejects decimal step values and falls back to step 1 (the paywall)", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=2.5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-welcome")).toBeNull();
  });

  it("rejects non-numeric step values and falls back to step 1 (the paywall)", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=foo");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
  });

  it("lets ?step=2&subscription=success land on Welcome past the paywall", async () => {
    // Simulates returning from a successful Stripe checkout. The highest
    // reached step before the redirect was 1 (the paywall). Without the
    // success-bypass, ceiling=min(1,2)=1 would clamp the user back onto the
    // paywall they just paid through.
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "1");
    currentSearchParams = new URLSearchParams("step=2&subscription=success");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("returns to SubscriptionStep on ?step=1&subscription=cancelled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "1");
    currentSearchParams = new URLSearchParams("step=1&subscription=cancelled");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
  });

  it("skips SubscriptionStep when the user is already on a paid tier", async () => {
    // Regression for paying users (admin-granted Pro, or accounts that
    // pre-date VISIT_COPILOT) being kicked through onboarding and asked to
    // pay again to escape. With ENABLE_PLATFORM_PAYMENT on and tier=PRO,
    // step 4 must render Preparing — not SubscriptionStep.
    mockFlagValue = true;
    mockSubscriptionTier = "PRO";
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=5 to preparingStep=4 for paid users", async () => {
    // For paid users the wizard is 3-step (preparingStep=4). A URL pointing
    // at the old 5-step preparingStep must clamp down to 4, not strand the
    // user above the ceiling.
    mockFlagValue = true;
    mockSubscriptionTier = "MAX";
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "5");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("still shows SubscriptionStep for NO_TIER users with payments enabled", async () => {
    mockFlagValue = true;
    mockSubscriptionTier = "NO_TIER";
    currentSearchParams = new URLSearchParams("step=1");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("waits for Supabase auth before initialising (no premature step lock)", async () => {
    // Regression: LD can resolve while auth is still loading. Without
    // gating on isUserLoading, isReady flips true (the !isLoggedIn branch
    // short-circuits), init runs against the pre-tier preparingStep (5)
    // and a returning user with highestStep=5 lands on currentStep=5.
    // When tier resolves to PRO, preparingStep becomes 4 — but the
    // hasInitialized guard blocks re-init, leaving currentStep=5 with
    // no matching step guard (blank page).
    mockFlagValue = true;
    mockSubscriptionTier = "PRO";
    mockSupabaseState = { isLoggedIn: false, isUserLoading: true };
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "5");
    render(<OnboardingPage />);
    // Nothing should render while auth is still loading.
    expect(screen.queryByTestId("step-welcome")).toBeNull();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
    // After init defers (auth not ready), currentStep stays at the
    // store default of 1 — no premature jump to 5.
    expect(useOnboardingWizardStore.getState().currentStep).toBe(1);
  });

  it("preserves form data on mount (zustand persist; no reset-on-init)", async () => {
    // Regression test for the 422 caused by init's old `reset()` wiping
    // name/role on every mount. With zustand persist, refreshing mid-wizard
    // must preserve what the user already typed.
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    useOnboardingWizardStore.setState({
      name: "Alice",
      role: "Engineering",
      painPoints: ["slow builds"],
    });
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    const state = useOnboardingWizardStore.getState();
    expect(state.name).toBe("Alice");
    expect(state.role).toBe("Engineering");
    expect(state.painPoints).toEqual(["slow builds"]);
  });
});
