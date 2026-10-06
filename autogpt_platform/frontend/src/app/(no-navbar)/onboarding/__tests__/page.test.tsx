import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import type { User } from "@/lib/auth/types";
import type { OnboardingWizardProgress } from "@/app/api/__generated__/models/onboardingWizardProgress";
import {
  installGtagShim,
  removeGtagShim,
} from "@/tests/integrations/gtag-shim";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";
import { progressStorageKey } from "../progress";
import { makeProgress } from "./progress-fixture";

vi.mock("../steps/RoleStep", () => ({
  RoleStep: () => (
    <button
      data-testid="step-role"
      onClick={() => useOnboardingWizardStore.getState().nextStep()}
    >
      Continue
    </button>
  ),
}));
vi.mock("../steps/PainPointsStep", () => ({
  PainPointsStep: () => (
    <button
      data-testid="step-painpoints"
      onClick={() => useOnboardingWizardStore.getState().nextStep()}
    >
      Continue
    </button>
  ),
}));
vi.mock("../steps/BrainDumpStep/BrainDumpStep", () => ({
  BrainDumpStep: () => (
    <button
      data-testid="step-braindump"
      onClick={() => useOnboardingWizardStore.getState().nextStep()}
    >
      Continue
    </button>
  ),
}));
vi.mock("../steps/ConnectStep/ConnectStep", () => ({
  ConnectStep: () => <div data-testid="step-connect" />,
}));
vi.mock("../steps/SubscriptionStep/SubscriptionStep", () => ({
  SubscriptionStep: () => <div data-testid="step-subscription" />,
}));
vi.mock("../steps/IntroStep/IntroStep", () => ({
  IntroStep: ({ slide }: { slide: string }) => (
    <button
      data-testid={`step-${slide}`}
      onClick={() => useOnboardingWizardStore.getState().nextStep()}
    >
      Continue
    </button>
  ),
}));
vi.mock("../steps/HireStep/HireStep", () => ({
  HireStep: () => (
    <button
      data-testid="step-hire"
      onClick={() => useOnboardingWizardStore.getState().nextStep()}
    >
      Continue
    </button>
  ),
}));
vi.mock("../steps/PreparingStep", () => ({
  PreparingStep: ({ onComplete }: { onComplete: () => void }) => (
    <button data-testid="step-preparing" onClick={onComplete}>
      Finish
    </button>
  ),
}));

let params = new URLSearchParams();
const routerReplace = vi.fn();
vi.mock("next/navigation", () => ({
  useRouter: () => ({
    replace: routerReplace,
    push: vi.fn(),
    refresh: vi.fn(),
  }),
  useSearchParams: () => params,
  usePathname: () => "/onboarding",
}));
let user: User | null;
let authLoading = false;
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: !!user,
    isUserLoading: authLoading,
    user,
    refreshSession: vi.fn(),
  }),
}));
let serverDraft: OnboardingWizardProgress | null = null;
let completed = false;
let revision = 0;
const { getState, patchState, completeStep, submitProfile } = vi.hoisted(
  () => ({
    getState: vi.fn(),
    patchState: vi.fn(),
    completeStep: vi.fn(),
    submitProfile: vi.fn(),
  }),
);
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: getState,
  patchV1UpdateOnboardingState: patchState,
  postV1CompleteOnboardingStep: completeStep,
  postV1SubmitOnboardingProfile: submitProfile,
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
}));
let tier = "NO_TIER";
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  getGetSubscriptionStatusQueryKey: () => ["/api/credits/subscription"],
  useGetSubscriptionStatus: (opts: {
    query: { select: (res: { status: number; data: unknown }) => unknown };
  }) => ({
    data: opts.query.select({ status: 200, data: { tier } }),
    isLoading: false,
  }),
}));
let trial = {
  ready: true,
  active: false,
  error: undefined as string | undefined,
  retry: vi.fn(),
};
vi.mock("@/services/trials/useTrialCheckoutReturn", () => ({
  useTrialCheckoutReturn: () => trial,
}));
let flags: Record<string, boolean> = {};
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {
    ENABLE_PLATFORM_PAYMENT: "payment",
    ONBOARDING_BRAIN_DUMP: "brain",
    ONBOARDING_EXPERT_TEAM: "team",
    HIRE_EXPERTS: "hire",
  },
  useGetFlag: (flag: string) => !!flags[flag],
  useFlagStatus: (flag: string) => ({ enabled: !!flags[flag], ready: true }),
}));
let local = false;
vi.mock("@/services/environment", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/environment")>();
  return {
    ...actual,
    environment: { ...actual.environment, isLocal: () => local },
  };
});
vi.mock("launchdarkly-react-client-sdk", () => ({
  useLDClient: () => ({ waitForInitialization: () => Promise.resolve() }),
}));

beforeEach(() => {
  user = {
    id: "u1",
    email: "reinier@example.com",
    user_metadata: { name: "Reinier Bot" },
  };
  params = new URLSearchParams();
  authLoading = false;
  tier = "NO_TIER";
  local = false;
  flags = { payment: true };
  completed = false;
  revision = 0;
  serverDraft = null;
  trial = { ready: true, active: false, error: undefined, retry: vi.fn() };
  localStorage.clear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  vi.clearAllMocks();
  getState.mockImplementation(() =>
    Promise.resolve({
      status: 200,
      data: {
        userId: user?.id,
        completedSteps: completed ? ["ONBOARDING_COMPLETE"] : [],
        wizardProgress: serverDraft,
        wizardRevision: revision,
      },
    }),
  );
  patchState.mockImplementation(
    ({ wizardProgress }: { wizardProgress: OnboardingWizardProgress }) => {
      serverDraft = wizardProgress;
      return Promise.resolve({
        status: 200,
        data: { userId: user?.id, wizardRevision: ++revision },
      });
    },
  );
  completeStep.mockResolvedValue({ status: 200 });
  submitProfile.mockResolvedValue({ status: 200 });
});
afterEach(() => {
  cleanup();
  removeGtagShim();
  vi.unstubAllEnvs();
});

function atPaywall() {
  serverDraft = makeProgress({
    currentStep: "subscription",
    completedSteps: ["role", "painPoints"],
  });
}

describe("onboarding before payment", () => {
  it("collects role and pain points before showing the card requirement", async () => {
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-role"));
    fireEvent.click(await screen.findByTestId("step-painpoints"));
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(completeStep).not.toHaveBeenCalled();
    expect(submitProfile).not.toHaveBeenCalled();
    fireEvent.click(screen.getByRole("button", { name: "Back" }));
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
  });

  it("keeps the brain dump and real hire step before the paywall", async () => {
    flags = { payment: true, team: true, brain: true, hire: true };
    render(<OnboardingPage />);
    for (const key of ["team", "autopilot", "role", "braindump", "hire"]) {
      fireEvent.click(await screen.findByTestId(`step-${key}`));
    }
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(useOnboardingWizardStore.getState().completedSteps).toContain(
      "hire",
    );
  });

  it("restores answers, hires and semantic progress in a fresh session", async () => {
    serverDraft = makeProgress({
      currentStep: "subscription",
      completedSteps: ["role", "painPoints"],
      otherRole: "Platform",
      selectedBilling: "yearly",
      hiredTemplateIds: ["maria"],
    });
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    expect(useOnboardingWizardStore.getState()).toMatchObject({
      role: "Engineering",
      otherRole: "Platform",
      painPoints: ["Reporting"],
      selectedBilling: "yearly",
      hiredTemplateIds: ["maria"],
    });
  });

  it.each(["subscription", "trial"])(
    "restores a cancelled %s checkout at the final paywall",
    async (kind) => {
      atPaywall();
      params = new URLSearchParams(`${kind}=cancelled`);
      render(<OnboardingPage />);
      expect(await screen.findByTestId("step-subscription")).toBeDefined();
      expect(useOnboardingWizardStore.getState().role).toBe("Engineering");
      expect(completeStep).not.toHaveBeenCalled();
    },
  );

  it.each(["subscription", "trial"])(
    "resumes after a verified %s checkout without replaying onboarding",
    async (kind) => {
      atPaywall();
      params = new URLSearchParams(`${kind}=success&step=preparing`);
      if (kind === "trial") trial.active = true;
      else tier = "PRO";
      render(<OnboardingPage />);
      expect(await screen.findByTestId("step-preparing")).toBeDefined();
      await waitFor(() =>
        expect(submitProfile).toHaveBeenCalledWith(
          {
            user_name: "Reinier",
            user_role: "Engineering",
            pain_points: ["Reporting"],
          },
          expect.objectContaining({ signal: expect.any(AbortSignal) }),
        ),
      );
      fireEvent.click(screen.getByTestId("step-preparing"));
      await waitFor(() =>
        expect(routerReplace).toHaveBeenCalledWith("/copilot"),
      );
      expect(completeStep).toHaveBeenCalledWith({
        step: "ONBOARDING_COMPLETE",
      });
    },
  );

  it("waits for trial confirmation before changing its return URL", async () => {
    atPaywall();
    params = new URLSearchParams("trial=success");
    trial.ready = false;
    const view = render(<OnboardingPage />);
    expect(
      screen.getByText("Confirming your trial and card setup…"),
    ).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalled();
    trial = { ...trial, ready: true, active: true };
    view.rerender(<OnboardingPage />);
    expect(await screen.findByTestId("step-preparing")).toBeDefined();
  });

  it("never trusts success query parameters or completed subscription drafts as entitlement", async () => {
    serverDraft = makeProgress({
      currentStep: "preparing",
      completedSteps: ["role", "painPoints", "subscription"],
    });
    params = new URLSearchParams("step=preparing&subscription=success");
    render(<OnboardingPage />);
    expect(screen.getByText("Confirming your subscription…")).toBeDefined();
    expect(completeStep).not.toHaveBeenCalled();
    expect(screen.queryByTestId("step-preparing")).toBeNull();
  });

  it("remaps a payment slot removed by a paid plan without skipping content", async () => {
    tier = "PRO";
    serverDraft = makeProgress({
      currentStep: "subscription",
      completedSteps: ["role"],
    });
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(screen.queryByTestId("step-subscription")).toBeNull();
  });

  it("restores the earliest incomplete step when optional intro steps become enabled", async () => {
    atPaywall();
    flags.team = true;
    flags.hire = true;
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-team")).toBeDefined();
  });

  it.each(["7", "2.5", "foo", "preparing"])(
    "ignores unsafe legacy or unreached progress %s",
    async (step) => {
      params = new URLSearchParams(`step=${step}`);
      sessionStorage.setItem("autogpt:onboarding-highest-step", "7");
      sessionStorage.setItem(
        "onboarding-wizard",
        JSON.stringify({ state: { role: "Another user" } }),
      );
      render(<OnboardingPage />);
      expect(await screen.findByTestId("step-role")).toBeDefined();
      expect(useOnboardingWizardStore.getState().role).toBe("");
    },
  );

  it("keeps self-host connection after personalization", async () => {
    flags.payment = false;
    local = true;
    serverDraft = makeProgress({
      currentStep: "connect",
      completedSteps: ["role", "painPoints"],
    });
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-connect")).toBeDefined();
  });

  it("holds onboarding while authentication is loading", () => {
    authLoading = true;
    render(<OnboardingPage />);
    expect(screen.queryByTestId("step-role")).toBeNull();
    expect(screen.getByRole("status").textContent).toBe("Loading your setup…");
    expect(getState).not.toHaveBeenCalled();
  });

  it("redirects completed users without rendering or resaving their draft", async () => {
    completed = true;
    render(<OnboardingPage />);
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
    expect(screen.queryByTestId("step-role")).toBeNull();
    expect(patchState).not.toHaveBeenCalled();
  });

  it("does not drop progress or redirect when completion fails", async () => {
    atPaywall();
    tier = "PRO";
    completeStep.mockRejectedValue(new Error("offline"));
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    expect(await screen.findByText(/couldn't finish setting up/)).toBeDefined();
    expect(routerReplace).not.toHaveBeenCalledWith("/copilot");
    expect(localStorage.getItem(progressStorageKey("u1"))).not.toBeNull();
  });
});

describe("draft recovery", () => {
  it("saves each step in the background without holding navigation", async () => {
    patchState.mockImplementation(() => new Promise(() => {}));
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-role"));
    fireEvent.click(await screen.findByTestId("step-painpoints"));
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    const localDraft = JSON.parse(
      localStorage.getItem(progressStorageKey("u1"))!,
    );
    expect(localDraft.progress.currentStep).toBe("subscription");
    expect(localDraft.pending).toBe(true);
  });

  it("recovers unsaved local answers over an older server snapshot", async () => {
    localStorage.setItem(
      progressStorageKey("u1"),
      JSON.stringify({
        pending: true,
        revision: 0,
        progress: makeProgress({
          currentStep: "painPoints",
          completedSteps: ["role"],
          role: "Design",
        }),
      }),
    );
    serverDraft = makeProgress({ role: "Old answer" });
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-painpoints")).toBeDefined();
    expect(useOnboardingWizardStore.getState().role).toBe("Design");
    await waitFor(() =>
      expect(patchState).toHaveBeenCalledWith(
        expect.objectContaining({
          wizardProgress: expect.objectContaining({ role: "Design" }),
        }),
        expect.anything(),
      ),
    );
  });

  it("surfaces failed loads instead of overwriting an inaccessible account draft", async () => {
    getState.mockRejectedValue(new Error("offline"));
    render(<OnboardingPage />);
    expect(await screen.findByText(/couldn't restore/)).toBeDefined();
    expect(screen.queryByTestId("step-role")).toBeNull();
    expect(patchState).not.toHaveBeenCalled();
  });

  it("surfaces save failures while allowing continuation", async () => {
    patchState.mockRejectedValue(new Error("offline"));
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-role"));
    expect(
      await screen.findByText(/couldn't save your progress/),
    ).toBeDefined();
    fireEvent.click(screen.getByTestId("step-painpoints"));
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
  });

  it("discards in-memory answers on account switch and never saves them for the next user", async () => {
    atPaywall();
    const view = render(<OnboardingPage />);
    expect(await screen.findByTestId("step-subscription")).toBeDefined();
    await act(async () => {
      await useOnboardingWizardStore.getState().flushProgress?.();
    });
    user = { id: "u2", email: "other@example.com", user_metadata: {} };
    serverDraft = null;
    patchState.mockClear();
    view.rerender(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(useOnboardingWizardStore.getState().role).toBe("");
    fireEvent.click(screen.getByTestId("step-role"));
    await act(async () => {
      await useOnboardingWizardStore.getState().flushProgress?.();
    });
    expect(patchState).toHaveBeenCalledWith(
      expect.objectContaining({
        wizardProgress: expect.objectContaining({ role: "" }),
      }),
      expect.anything(),
    );
  });
});

it("holds a paid-checkout return until its webhook updates entitlement", async () => {
  atPaywall();
  params = new URLSearchParams("subscription=success&step=preparing");
  const view = render(<OnboardingPage />);
  expect(screen.getByText("Confirming your subscription…")).toBeDefined();
  expect(screen.queryByTestId("step-subscription")).toBeNull();
  tier = "PRO";
  view.rerender(<OnboardingPage />);
  expect(await screen.findByTestId("step-preparing")).toBeDefined();
  expect(useOnboardingWizardStore.getState().role).toBe("Engineering");
});

it("ignores a late draft response from an account that has been switched away", async () => {
  let release: ((value: unknown) => void) | undefined;
  getState.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  );
  const view = render(<OnboardingPage />);
  user = { id: "u2", email: "other@example.com", user_metadata: {} };
  view.rerender(<OnboardingPage />);
  expect(await screen.findByTestId("step-role")).toBeDefined();
  await act(async () => {
    release?.({
      status: 200,
      data: {
        userId: "u1",
        completedSteps: [],
        wizardRevision: 0,
        wizardProgress: makeProgress({ role: "Private old answer" }),
      },
    });
  });
  expect(useOnboardingWizardStore.getState().role).toBe("");
  expect(useOnboardingWizardStore.getState().userID).toBe("u2");
});

it("asks before replacing a stale local draft with another session's progress", async () => {
  localStorage.setItem(
    progressStorageKey("u1"),
    JSON.stringify({
      pending: true,
      revision: 0,
      progress: makeProgress({ role: "Local answer" }),
    }),
  );
  serverDraft = makeProgress({ role: "Newer answer" });
  revision = 2;
  render(<OnboardingPage />);
  fireEvent.click(
    await screen.findByRole("button", { name: "Reload latest progress" }),
  );
  expect(await screen.findByTestId("step-role")).toBeDefined();
  expect(useOnboardingWizardStore.getState().role).toBe("Newer answer");
});

it("recognizes a successful save whose response was lost before navigation", async () => {
  const draft = makeProgress({
    currentStep: "painPoints",
    completedSteps: ["role"],
  });
  localStorage.setItem(
    progressStorageKey("u1"),
    JSON.stringify({ pending: true, revision: 0, progress: draft }),
  );
  serverDraft = draft;
  revision = 1;
  render(<OnboardingPage />);
  expect(await screen.findByTestId("step-painpoints")).toBeDefined();
  expect(
    screen.queryByRole("button", { name: "Reload latest progress" }),
  ).toBeNull();
  await act(async () => {
    await useOnboardingWizardStore.getState().flushProgress?.();
  });
  expect(patchState).not.toHaveBeenCalled();
  expect(
    JSON.parse(localStorage.getItem(progressStorageKey("u1"))!),
  ).toMatchObject({
    pending: false,
    revision: 1,
  });
  await act(async () => {
    useOnboardingWizardStore.getState().setRole("Design");
    await useOnboardingWizardStore.getState().flushProgress?.();
  });
  expect(patchState).toHaveBeenCalledWith(
    expect.objectContaining({ wizardRevision: 1 }),
    expect.anything(),
  );
});

it("surfaces a revision conflict without overwriting newer server answers", async () => {
  patchState.mockRejectedValue({ status: 409 });
  render(<OnboardingPage />);
  fireEvent.click(await screen.findByTestId("step-role"));
  expect(
    await screen.findByRole("button", { name: "Reload latest progress" }),
  ).toBeDefined();
  expect(screen.queryByTestId("step-role")).toBeNull();
  expect(
    JSON.parse(localStorage.getItem(progressStorageKey("u1"))!).pending,
  ).toBe(true);
});

it("reports onboarding completion to Google Ads once after confirmed completion", async () => {
  vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
  vi.stubEnv(
    "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
    "onboarding_complete=OC",
  );
  const calls = installGtagShim();
  atPaywall();
  tier = "PRO";
  render(<OnboardingPage />);
  const finish = await screen.findByTestId("step-preparing");
  fireEvent.click(finish);
  fireEvent.click(finish);
  await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
  expect(calls.filter((call) => call[1] === "conversion")).toHaveLength(1);
  expect(completeStep).toHaveBeenCalledOnce();
});

it("reports no Ads completion when the server rejects completion", async () => {
  vi.stubEnv("NEXT_PUBLIC_GOOGLE_ADS_ID", "AW-123");
  vi.stubEnv(
    "NEXT_PUBLIC_GOOGLE_ADS_CONVERSION_LABELS",
    "onboarding_complete=OC",
  );
  const calls = installGtagShim();
  atPaywall();
  tier = "PRO";
  completeStep.mockRejectedValue(new Error("offline"));
  render(<OnboardingPage />);
  fireEvent.click(await screen.findByTestId("step-preparing"));
  expect(await screen.findByText(/couldn't finish setting up/)).toBeDefined();
  expect(calls.filter((call) => call[1] === "conversion")).toEqual([]);
});

it.each(["", "Other"])(
  "returns an incomplete %s role to editing before completing onboarding",
  async (role) => {
    serverDraft = makeProgress({
      currentStep: "preparing",
      completedSteps: ["role", "painPoints"],
      role,
      otherRole: "   ",
    });
    tier = "PRO";
    render(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    expect(submitProfile).not.toHaveBeenCalled();
    expect(completeStep).not.toHaveBeenCalled();
    act(() => useOnboardingWizardStore.getState().setRole("Design"));
    fireEvent.click(screen.getByTestId("step-role"));
    fireEvent.click(await screen.findByTestId("step-painpoints"));
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
    expect(submitProfile).toHaveBeenCalledWith(
      expect.objectContaining({ user_role: "Design" }),
      expect.anything(),
    );
    expect(completeStep).toHaveBeenCalledOnce();
  },
);

it("rejects a response for another signed-in account without hydrating or caching it", async () => {
  getState.mockResolvedValue({
    status: 200,
    data: {
      userId: "another-user",
      wizardRevision: 0,
      completedSteps: [],
      wizardProgress: makeProgress({ role: "Private answer" }),
    },
  });
  render(<OnboardingPage />);
  expect(await screen.findByText(/signed-in account changed/)).toBeDefined();
  expect(useOnboardingWizardStore.getState().role).toBe("");
  expect(localStorage.getItem(progressStorageKey("u1"))).toBeNull();
  expect(patchState).not.toHaveBeenCalled();
});

describe("profile saving before completion", () => {
  it("waits for the in-flight profile save before completing or clearing the draft", async () => {
    atPaywall();
    tier = "PRO";
    let release: ((value: unknown) => void) | undefined;
    submitProfile.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await act(async () => {
      await useOnboardingWizardStore.getState().flushProgress?.();
    });
    expect(submitProfile).toHaveBeenCalledOnce();
    expect(completeStep).not.toHaveBeenCalled();
    expect(routerReplace).not.toHaveBeenCalledWith("/copilot");
    expect(localStorage.getItem(progressStorageKey("u1"))).not.toBeNull();
    await act(async () => release?.({ status: 200 }));
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/copilot"));
    expect(submitProfile).toHaveBeenCalledOnce();
    expect(completeStep).toHaveBeenCalledOnce();
    expect(localStorage.getItem(progressStorageKey("u1"))).toBeNull();
  });

  it.each(["network", "http"])(
    "keeps a %s profile-save failure visible and retries before completion",
    async (failure) => {
      atPaywall();
      tier = "PRO";
      if (failure === "network")
        submitProfile.mockRejectedValue(new Error("offline"));
      else submitProfile.mockResolvedValue({ status: 503 });
      render(<OnboardingPage />);
      expect(
        await screen.findByText(/couldn't save your profile/i),
      ).toBeDefined();
      fireEvent.click(screen.getByTestId("step-preparing"));
      await waitFor(() => expect(submitProfile).toHaveBeenCalledTimes(2));
      expect(completeStep).not.toHaveBeenCalled();
      expect(routerReplace).not.toHaveBeenCalledWith("/copilot");
      expect(localStorage.getItem(progressStorageKey("u1"))).not.toBeNull();
      submitProfile.mockResolvedValue({ status: 200 });
      fireEvent.click(screen.getByRole("button", { name: /try again/i }));
      await waitFor(() =>
        expect(routerReplace).toHaveBeenCalledWith("/copilot"),
      );
      expect(submitProfile).toHaveBeenCalledTimes(3);
      expect(completeStep).toHaveBeenCalledOnce();
    },
  );

  it("does not complete either account when an old profile save resolves after switching users", async () => {
    atPaywall();
    tier = "PRO";
    let release: ((value: unknown) => void) | undefined;
    submitProfile.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          release = resolve;
        }),
    );
    const view = render(<OnboardingPage />);
    fireEvent.click(await screen.findByTestId("step-preparing"));
    await act(async () => {
      await useOnboardingWizardStore.getState().flushProgress?.();
    });
    user = { id: "u2", email: "other@example.com", user_metadata: {} };
    serverDraft = null;
    view.rerender(<OnboardingPage />);
    expect(await screen.findByTestId("step-role")).toBeDefined();
    await act(async () => release?.({ status: 200 }));
    expect(completeStep).not.toHaveBeenCalled();
    expect(routerReplace).not.toHaveBeenCalledWith("/copilot");
    expect(useOnboardingWizardStore.getState().userID).toBe("u2");
    expect(localStorage.getItem(progressStorageKey("u1"))).not.toBeNull();
  });
});

it("lets a new account finish while the previous account's profile save is outstanding", async () => {
  atPaywall();
  tier = "PRO";
  let release: ((value: unknown) => void) | undefined;
  submitProfile.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        release = resolve;
      }),
  );
  const view = render(<OnboardingPage />);
  fireEvent.click(await screen.findByTestId("step-preparing"));
  await act(async () => {
    await useOnboardingWizardStore.getState().flushProgress?.();
  });
  user = { id: "u2", email: "other@example.com", user_metadata: {} };
  serverDraft = makeProgress({
    currentStep: "preparing",
    completedSteps: ["role", "painPoints"],
    role: "Design",
  });
  view.rerender(<OnboardingPage />);
  await waitFor(() => expect(submitProfile).toHaveBeenCalledTimes(2));
  fireEvent.click(await screen.findByTestId("step-preparing"));
  await waitFor(() => expect(completeStep).toHaveBeenCalledOnce());
  expect(routerReplace).toHaveBeenCalledWith("/copilot");
  await act(async () => release?.({ status: 200 }));
  expect(completeStep).toHaveBeenCalledOnce();
  expect(screen.queryByText(/couldn't save your profile/i)).toBeNull();
});
