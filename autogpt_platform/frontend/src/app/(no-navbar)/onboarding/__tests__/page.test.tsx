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

beforeEach(() => {
  currentSearchParams = new URLSearchParams();
  mockFlagValue = false;
  routerReplace.mockClear();
  useOnboardingWizardStore.getState().reset();
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
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("renders SubscriptionStep at step 4 when payments are enabled", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("treats step 5 as Preparing when payments are enabled", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
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
});
