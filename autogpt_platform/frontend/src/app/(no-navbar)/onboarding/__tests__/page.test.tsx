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

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: () => ({ isLoggedIn: true, isUserLoading: false, user: null }),
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

  it("renders SubscriptionStep at step 4 when payments are enabled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
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
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    // Highest reached is 2 (RoleStep), so manually editing the URL to step=5
    // should land the user back on step 2, not let them skip ahead.
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("resumes from the highest reached step when ?step= is omitted", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    // No step param — user navigated to /onboarding directly.
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(screen.queryByTestId("step-welcome")).toBeNull();
  });

  it("rejects decimal step values and falls back to step 1", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=2.5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
    expect(screen.queryByTestId("step-role")).toBeNull();
  });

  it("rejects non-numeric step values and falls back to step 1", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=foo");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-welcome")).toBeDefined();
  });

  it("lets ?step=5&subscription=success bypass the highestStep ceiling", async () => {
    // Simulates returning from a successful Stripe checkout. The highest
    // reached step before the redirect was 4 (SubscriptionStep). Without the
    // success-bypass, ceiling=min(4,5)=4 would clamp the user back to step 4.
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=5&subscription=success");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("returns to SubscriptionStep on ?step=4&subscription=cancelled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4&subscription=cancelled");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(useOnboardingWizardStore.getState().selectedPlan).toBeNull();
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
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    const state = useOnboardingWizardStore.getState();
    expect(state.name).toBe("Alice");
    expect(state.role).toBe("Engineering");
    expect(state.painPoints).toEqual(["slow builds"]);
  });
});
