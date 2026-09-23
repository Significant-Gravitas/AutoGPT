import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import type { User } from "@/lib/auth/types";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";

vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => <div data-testid="step-role" />,
}));
vi.mock("../steps/PainPointsStep", () => ({
  PainPointsStep: () => <div data-testid="step-painpoints" />,
}));
vi.mock("../steps/ConnectStep/ConnectStep", () => ({
  ConnectStep: () => <div data-testid="step-connect" />,
}));

vi.mock("../steps/SubscriptionStep/SubscriptionStep", () => ({
  SubscriptionStep: () => <div data-testid="step-subscription" />,
}));
vi.mock("../steps/IntroStep/IntroStep", () => ({
  IntroStep: ({ slide }: { slide: string }) => (
    <div data-testid={`step-${slide}`} />
  ),
}));
vi.mock("../steps/HireStep/HireStep", () => ({
  HireStep: () => <div data-testid="step-hire" />,
}));
vi.mock("../steps/PreparingStep", () => ({
  PreparingStep: ({ onComplete }: { onComplete: () => void }) => (
    <button data-testid="step-preparing" onClick={onComplete} />
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

let mockAuthState = { isLoggedIn: true, isUserLoading: false };
let mockUser: User | null = null;
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    ...mockAuthState,
    user: mockUser,
    refreshSession: vi.fn(() => Promise.resolve({ user: null })),
  }),
}));

let mockCompletedSteps: string[] = [];
const { completeOnboardingStep, submitOnboardingProfile } = vi.hoisted(() => ({
  completeOnboardingStep: vi.fn(() => Promise.resolve({ status: 200 })),
  submitOnboardingProfile: vi.fn(() => Promise.resolve({ status: 200 })),
}));
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: () =>
    Promise.resolve({
      status: 200,
      data: { completedSteps: mockCompletedSteps },
    }),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: completeOnboardingStep,
  postV1SubmitOnboardingProfile: submitOnboardingProfile,
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

// Answers per-flag rather than returning one shared boolean: these tests
// only mean to toggle the paywall, and a blanket `true` would also switch
// on every other gated flag — including the brain dump, which replaces
// the pillbox step this file asserts on.
let mockFlagValue = false;
let mockExpertTeamEnabled = false;
let mockHireExpertsEnabled = true;
let mockBrainDumpEnabled = false;
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ENABLE_PLATFORM_PAYMENT: "ENABLE_PLATFORM_PAYMENT",
    ONBOARDING_BRAIN_DUMP: "ONBOARDING_BRAIN_DUMP",
    ONBOARDING_EXPERT_TEAM: "ONBOARDING_EXPERT_TEAM",
    HIRE_EXPERTS: "HIRE_EXPERTS",
  },
  useGetFlag: (flag: string) => {
    if (flag === "ENABLE_PLATFORM_PAYMENT") return mockFlagValue;
    if (flag === "ONBOARDING_EXPERT_TEAM") return mockExpertTeamEnabled;
    if (flag === "HIRE_EXPERTS") return mockHireExpertsEnabled;
    if (flag === "ONBOARDING_BRAIN_DUMP") return mockBrainDumpEnabled;
    return false;
  },
}));

// The brain dump step is the real one when its flag is on; the page tests
// only care which slot it occupies.
vi.mock("../steps/BrainDumpStep/BrainDumpStep", () => ({
  BrainDumpStep: () => <div data-testid="step-braindump" />,
}));

vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, isLocal: () => mockIsLocal },
  };
});

vi.mock("launchdarkly-react-client-sdk", () => ({
  useLDClient: () => ({
    waitForInitialization: () => Promise.resolve(),
  }),
}));

const STEP_STORAGE_KEY = "autogpt:onboarding-highest-step";
let mockIsLocal = false;

beforeEach(() => {
  currentSearchParams = new URLSearchParams();
  mockFlagValue = false;
  mockExpertTeamEnabled = false;
  mockHireExpertsEnabled = true;
  mockBrainDumpEnabled = false;
  mockIsLocal = false;
  mockSubscriptionTier = "NO_TIER";
  mockAuthState = { isLoggedIn: true, isUserLoading: false };
  mockUser = null;
  mockCompletedSteps = [];
  routerReplace.mockClear();
  completeOnboardingStep.mockReset();
  completeOnboardingStep.mockResolvedValue({ status: 200 });
  submitOnboardingProfile.mockClear();
  useOnboardingWizardStore.getState().reset();
  window.sessionStorage.removeItem(STEP_STORAGE_KEY);
});

afterEach(() => {
  cleanup();
  // Not at the end of each test body: an assertion that throws above would
  // leak the shim and NEXT_PUBLIC_GOOGLE_ADS_ID into every later test here.
  removeGtagShim();
  vi.unstubAllEnvs();
});

describe("OnboardingPage — intro steps", () => {
  it("starts on the Team step when the expert team is on", async () => {
    mockExpertTeamEnabled = true;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-team")).toBeDefined();
    expect(screen.queryByTestId("step-role")).toBeNull();
  });

  it("walks Team → Meet Otto → Role, tracking each as a step", async () => {
    mockExpertTeamEnabled = true;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-team")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-autopilot")).toBeDefined();
    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/onboarding?step=2", {
        scroll: false,
      });
    });

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
    expect(window.sessionStorage.getItem(STEP_STORAGE_KEY)).toBe("3");
  });

  it("lets Back return from Role to Meet Otto", async () => {
    mockExpertTeamEnabled = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await screen.findByTestId("step-autopilot")).toBeDefined();
  });

  it("hides Back on the Team step", async () => {
    mockExpertTeamEnabled = true;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-team")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
  });

  it("resumes on Meet Otto when that is the highest step reached", async () => {
    mockExpertTeamEnabled = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-autopilot")).toBeDefined();
    expect(screen.queryByTestId("step-team")).toBeNull();
  });

  it("puts the paywall before the intro steps and Preparing last", async () => {
    mockExpertTeamEnabled = true;
    mockFlagValue = true;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-team")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().goToStep(6));
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
  });

  it("has no intro steps when the expert team flag is off", async () => {
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-team")).toBeNull();
  });
});

describe("OnboardingPage — hire step", () => {
  it("follows the brain dump with the hire step, then Preparing", async () => {
    mockExpertTeamEnabled = true;
    mockBrainDumpEnabled = true;
    mockFlagValue = true;
    // Paywall 1, Team 2, Meet Otto 3, Role 4, Brain dump 5, Hire 6,
    // Preparing 7.
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "5");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-braindump")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-hire")).toBeDefined();
    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/onboarding?step=6", {
        scroll: false,
      });
    });

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
  });

  it("lets Back return from the hire step to the brain dump", async () => {
    mockExpertTeamEnabled = true;
    mockBrainDumpEnabled = true;
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "6");
    currentSearchParams = new URLSearchParams("step=6");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-hire")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await screen.findByTestId("step-braindump")).toBeDefined();
  });

  it("has no hire step without the brain dump it reads from", async () => {
    mockExpertTeamEnabled = true;
    mockFlagValue = true;
    // Paywall 1, Team 2, Meet Otto 3, Role 4, Pain points 5, Preparing 6.
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "5");
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-hire")).toBeNull();
  });

  it("has no hire step when the expert team is off, even with the dump on", async () => {
    mockBrainDumpEnabled = true;
    mockFlagValue = true;
    // Paywall 1, Role 2, Brain dump 3, Preparing 4.
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-braindump")).toBeDefined();

    act(() => useOnboardingWizardStore.getState().nextStep());
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-hire")).toBeNull();
  });
});

describe("OnboardingPage — flag-gated SubscriptionStep", () => {
  it("renders Role at step 1 by default in flag-off mode", async () => {
    mockFlagValue = false;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=5 to step 1 when payments are gated off", async () => {
    mockFlagValue = false;
    currentSearchParams = new URLSearchParams("step=5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("treats step 3 as Preparing when payments are gated off", async () => {
    mockFlagValue = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("renders SubscriptionStep first (step 1) when payments are enabled", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=1");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-role")).toBeNull();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("renders Role at step 2 when payments are enabled — after the paywall", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
    currentSearchParams = new URLSearchParams("step=2");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("treats step 4 as Preparing when payments are enabled", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=4 to the user's highest reached step (no fast-forward)", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    // Highest reached is 3 (pain points), so manually editing the URL to
    // step=4 should land the user back on step 3, not let them skip ahead.
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("resumes from the highest reached step when ?step= is omitted", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    // No step param — user navigated to /onboarding directly.
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("rejects decimal step values and falls back to step 1 (paywall)", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=2.5");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-role")).toBeNull();
  });

  it("rejects non-numeric step values and falls back to step 1 (paywall)", async () => {
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=foo");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
  });

  it("lets ?step=2&subscription=success land on Role past the paywall", async () => {
    // Simulates returning from a successful Stripe checkout. The highest
    // reached step before the redirect was 1 (the paywall). Without the
    // success-bypass, ceiling=min(1,2)=1 would clamp the user back onto the
    // paywall they just paid through.
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "1");
    currentSearchParams = new URLSearchParams("step=2&subscription=success");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("does not let a subscription=success return skip past the step after the paywall", async () => {
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "1");
    currentSearchParams = new URLSearchParams("step=4&subscription=success");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
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
    // pre-date ONBOARDING_COMPLETE) being kicked through onboarding and asked to
    // pay again to escape. With ENABLE_PLATFORM_PAYMENT on and tier=PRO,
    // step 3 must render Preparing — not SubscriptionStep.
    mockFlagValue = true;
    mockSubscriptionTier = "PRO";
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("clamps ?step=4 to preparingStep=3 for paid users", async () => {
    // For paid users the wizard is 2-step (preparingStep=3). A URL pointing
    // at the paywall layout's preparingStep must clamp down to 3, not strand
    // the user above the ceiling.
    mockFlagValue = true;
    mockSubscriptionTier = "MAX";
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
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
    expect(screen.queryByTestId("step-role")).toBeNull();
  });

  it("waits for auth before initialising (no premature step lock)", async () => {
    // Regression: LD can resolve while auth is still loading. Without
    // gating on isUserLoading, isReady flips true (the !isLoggedIn branch
    // short-circuits), init runs against the pre-tier preparingStep (5)
    // and a returning user with highestStep=5 lands on currentStep=5.
    // When tier resolves to PRO, preparingStep becomes 4 — but the
    // hasInitialized guard blocks re-init, leaving currentStep=5 with
    // no matching step guard (blank page).
    mockFlagValue = true;
    mockSubscriptionTier = "PRO";
    mockAuthState = { isLoggedIn: false, isUserLoading: true };
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

  it("submits the profile under the account's name on reaching Preparing", async () => {
    // The wizard no longer asks for a name, so the profile carries what the
    // account already knows.
    mockFlagValue = false;
    mockUser = {
      id: "u1",
      email: "reinier@example.com",
      user_metadata: { name: "Reinier Bot" },
    };
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    useOnboardingWizardStore.setState({
      role: "Engineering",
      painPoints: ["slow builds"],
    });
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    await waitFor(() => {
      expect(submitOnboardingProfile).toHaveBeenCalledWith({
        user_name: "Reinier",
        user_role: "Engineering",
        pain_points: ["slow builds"],
      });
    });
  });

  it("does not submit a profile when no role was chosen", async () => {
    mockFlagValue = false;
    mockUser = { id: "u1", email: "reinier@example.com", user_metadata: {} };
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(submitOnboardingProfile).not.toHaveBeenCalled();
  });

  it("redirects straight to /copilot when onboarding is already complete", async () => {
    mockCompletedSteps = ["ONBOARDING_COMPLETE"];
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    render(<OnboardingPage />);
    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/copilot");
    });
    // The wizard never renders and the resume ceiling is cleared.
    expect(screen.queryByTestId("step-role")).toBeNull();
    expect(window.sessionStorage.getItem(STEP_STORAGE_KEY)).toBeNull();
  });

  it("marks ONBOARDING_COMPLETE and redirects when Preparing finishes", async () => {
    mockFlagValue = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/copilot");
    });
    expect(completeOnboardingStep).toHaveBeenCalledWith({
      step: "ONBOARDING_COMPLETE",
    });
    expect(window.sessionStorage.getItem(STEP_STORAGE_KEY)).toBeNull();
  });

  it("reports onboarding_complete to Google Ads when Preparing finishes", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
      "onboarding_complete=OC",
    );
    const gtagCalls = installGtagShim();
    mockFlagValue = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await waitFor(() => {
      expect(routerReplace).toHaveBeenCalledWith("/copilot");
    });

    expect(gtagCalls).toContainEqual([
      "event",
      "conversion",
      { send_to: "AW-123/OC" },
    ]);
  });

  it("reports no onboarding_complete when the API never confirms", async () => {
    vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
    vi.stubEnv(
      "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
      "onboarding_complete=OC",
    );
    const gtagCalls = installGtagShim();
    // All three attempts fail: the user still lands on the copilot, but the
    // backend never recorded the milestone, so nothing converted.
    completeOnboardingStep.mockRejectedValue(new Error("500"));
    mockFlagValue = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await waitFor(
      () => {
        expect(routerReplace).toHaveBeenCalledWith("/copilot");
      },
      { timeout: 5000 },
    );

    expect(completeOnboardingStep).toHaveBeenCalledTimes(3);
    expect(gtagCalls.filter((call) => call[1] === "conversion")).toEqual([]);
  }, 10000);

  it("preserves form data on mount (zustand persist; no reset-on-init)", async () => {
    // Regression test for the 422 caused by init's old `reset()` wiping
    // the role on every mount. With zustand persist, refreshing mid-wizard
    // must preserve what the user already chose.
    mockFlagValue = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    useOnboardingWizardStore.setState({
      role: "Engineering",
      painPoints: ["slow builds"],
    });
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    const state = useOnboardingWizardStore.getState();
    expect(state.role).toBe("Engineering");
    expect(state.painPoints).toEqual(["slow builds"]);
  });
});

describe("OnboardingPage — self-host closes with the connection", () => {
  it("renders Role first on a self-host install", async () => {
    mockIsLocal = true;
    currentSearchParams = new URLSearchParams("step=1");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(screen.queryByTestId("step-connect")).toBeNull();
  });

  it("puts ConnectStep last (step 3), after the profile", async () => {
    mockIsLocal = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-connect")).toBeDefined();
    expect(screen.queryByTestId("step-painpoints")).toBeNull();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("treats step 4 as Preparing on self-host", async () => {
    mockIsLocal = true;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "4");
    currentSearchParams = new URLSearchParams("step=4");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-connect")).toBeNull();
  });

  it("does not insert it on cloud", async () => {
    mockIsLocal = false;
    window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
    currentSearchParams = new URLSearchParams("step=3");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
    expect(screen.queryByTestId("step-connect")).toBeNull();
  });

  it("yields to the paywall rather than showing both model steps", async () => {
    // Payments and self-host are mutually exclusive in practice; if a
    // deployment ever had both, only one step can take the slot.
    mockIsLocal = true;
    mockFlagValue = true;
    currentSearchParams = new URLSearchParams("step=1");
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(screen.queryByTestId("step-connect")).toBeNull();
  });
});

it("never offers Back into the paywall, with or without the intro", async () => {
  mockFlagValue = true;
  window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
  currentSearchParams = new URLSearchParams("step=2");
  render(<OnboardingPage />);
  expect(await screen.findByTestId("step-role")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
  act(() => useOnboardingWizardStore.getState().nextStep());
  expect(await screen.findByTestId("step-painpoints")).toBeDefined();
  fireEvent.click(screen.getByRole("button", { name: "Back" }));
  expect(await screen.findByTestId("step-role")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
});

it("hides Back on the first intro after the paywall", async () => {
  mockFlagValue = true;
  mockExpertTeamEnabled = true;
  window.sessionStorage.setItem(STEP_STORAGE_KEY, "2");
  currentSearchParams = new URLSearchParams("step=2");
  render(<OnboardingPage />);
  expect(await screen.findByTestId("step-team")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Back" })).toBeNull();
});

it("skips both intro and hire when hiring is disabled independently of the team flag", async () => {
  mockExpertTeamEnabled = true;
  mockHireExpertsEnabled = false;
  mockBrainDumpEnabled = true;
  render(<OnboardingPage />);
  expect(await screen.findByTestId("step-role")).toBeDefined();
  expect(screen.queryByTestId("step-team")).toBeNull();
  act(() => useOnboardingWizardStore.getState().nextStep());
  expect(await screen.findByTestId("step-braindump")).toBeDefined();
  act(() => useOnboardingWizardStore.getState().nextStep());
  expect(await screen.findByTestId("step-preparing")).toBeDefined();
  expect(screen.queryByTestId("step-hire")).toBeNull();
});

it("submits the profile once if account data arrives after Preparing", async () => {
  window.sessionStorage.setItem(STEP_STORAGE_KEY, "3");
  currentSearchParams = new URLSearchParams("step=3");
  useOnboardingWizardStore.setState({
    role: "Engineering",
    painPoints: ["slow builds"],
  });
  const { rerender } = render(<OnboardingPage />);
  expect(await screen.findByTestId("step-preparing")).toBeDefined();
  expect(submitOnboardingProfile).not.toHaveBeenCalled();
  mockUser = {
    id: "u1",
    email: "reinier@example.com",
    user_metadata: { name: "Reinier Bot" },
  };
  rerender(<OnboardingPage />);
  await waitFor(() =>
    expect(submitOnboardingProfile).toHaveBeenCalledWith({
      user_name: "Reinier",
      user_role: "Engineering",
      pain_points: ["slow builds"],
    }),
  );
  mockUser = { ...mockUser };
  rerender(<OnboardingPage />);
  expect(submitOnboardingProfile).toHaveBeenCalledTimes(1);
});

it.each([1, 2])(
  "preserves reached step %s on a checkout return when payment is disabled",
  async (highest) => {
    window.sessionStorage.setItem(STEP_STORAGE_KEY, String(highest));
    currentSearchParams = new URLSearchParams("step=4&subscription=success");
    render(<OnboardingPage />);
    expect(
      await screen.findByTestId(
        highest === 1 ? "step-role" : "step-painpoints",
      ),
    ).toBeDefined();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  },
);
