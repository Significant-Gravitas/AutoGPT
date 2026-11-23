import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { getCurrentUser, validateSession } from "@/lib/auth/actions";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { getServerSession } from "@/lib/auth/server/getServerSession";
import type { User } from "@/lib/auth/types";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import OnboardingLayout from "../layout";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";

vi.mock("next/navigation", () => ({
  useRouter: () => ({ replace: vi.fn(), push: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/onboarding",
}));

// The browser's session check: each call waits until the test answers it.
vi.mock("@/lib/auth/actions", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/auth/actions")>()),
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
}));

vi.mock("../steps/IntroStep/IntroStep", () => ({
  IntroStep: ({ slide }: { slide: string }) => (
    <div data-testid={`step-${slide}`} />
  ),
}));

const { getState } = vi.hoisted(() => ({ getState: vi.fn() }));
vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1OnboardingState: getState,
  patchV1UpdateOnboardingState: vi.fn(),
  postV1CompleteOnboardingStep: vi.fn(),
  postV1SubmitOnboardingProfile: vi.fn(),
  getV1CheckIfOnboardingIsCompleted: () =>
    Promise.resolve({ status: 200, data: false }),
}));
vi.mock("@/app/api/__generated__/endpoints/credits/credits", () => ({
  getGetSubscriptionStatusQueryKey: () => ["/api/credits/subscription"],
  useGetSubscriptionStatus: (opts: {
    query: { select: (res: { status: number; data: unknown }) => unknown };
  }) => ({
    data: opts.query.select({ status: 200, data: { tier: "NO_TIER" } }),
    isLoading: false,
  }),
}));
vi.mock("@/services/trials/useTrialCheckoutReturn", () => ({
  useTrialCheckoutReturn: () => ({
    ready: true,
    active: false,
    error: undefined,
    retry: vi.fn(),
  }),
}));
vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: {},
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));
vi.mock("@/hooks/usePlatformPayment", () => ({
  usePlatformPayment: () => ({ isPaymentEnabled: false, isResolved: true }),
}));

const USER: User = {
  id: "user-1",
  email: "pat@example.com",
  role: "authenticated",
  user_metadata: { name: "Pat" },
};

// What Better Auth answers the layout for a request with a live session.
const SESSION = {
  user: {
    id: USER.id,
    email: USER.email,
    emailVerified: true,
    name: "Pat",
    role: "user",
    createdAt: new Date("2026-01-01T00:00:00Z"),
  },
  session: { id: "session-1", userId: USER.id },
};

const SIGNED_OUT = { user: null, isValid: false };

// Checks a test left unanswered, answered after it so they can't hold the
// store's one-check slot into the next test.
const unanswered: Array<() => void> = [];

function deferred<T>(answerAfterTest: T) {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((done) => {
    resolve = done;
  });
  unanswered.push(() => resolve(answerAfterTest));
  return { promise, resolve };
}

async function onboardingPage() {
  return render(await OnboardingLayout({ children: <OnboardingPage /> }));
}

describe("/onboarding first paint", () => {
  beforeEach(() => {
    useAuthStore.getState().cleanup();
    useAuthStore.setState(useAuthStore.getInitialState(), true);
    useOnboardingWizardStore.getState().reset();
    localStorage.clear();
    sessionStorage.clear();
    vi.mocked(getServerSession).mockResolvedValue(null);
    vi.mocked(getCurrentUser).mockResolvedValue({ user: null });
    vi.mocked(validateSession).mockResolvedValue(SIGNED_OUT);
    getState.mockImplementation(() =>
      Promise.resolve({
        status: 200,
        data: {
          userId: USER.id,
          completedSteps: [],
          wizardProgress: null,
          wizardRevision: 0,
        },
      }),
    );
  });

  afterEach(async () => {
    for (const answer of unanswered.splice(0)) answer();
    await waitFor(() =>
      expect(useAuthStore.getState().isValidating).toBe(false),
    );
    vi.clearAllMocks();
  });

  test("with the session the server read, the wizard loads before the browser's session check answers", async () => {
    vi.mocked(getServerSession).mockResolvedValue(
      SESSION as unknown as Awaited<ReturnType<typeof getServerSession>>,
    );
    const check =
      deferred<Awaited<ReturnType<typeof validateSession>>>(SIGNED_OUT);
    vi.mocked(validateSession).mockReturnValue(check.promise);

    await onboardingPage();

    expect(await screen.findByTestId("step-team")).toBeDefined();
    expect(getState).toHaveBeenCalled();
    expect(getCurrentUser).not.toHaveBeenCalled();
    // The check still runs, in the background.
    expect(validateSession).toHaveBeenCalledWith("/onboarding");

    check.resolve({ user: USER, isValid: true });
    await waitFor(() =>
      expect(useAuthStore.getState().isValidating).toBe(false),
    );
    expect(screen.getByTestId("step-team")).toBeDefined();
  });

  test("without a server session, the wizard waits for the browser's session check as before", async () => {
    const lookup = deferred<Awaited<ReturnType<typeof getCurrentUser>>>({
      user: null,
    });
    vi.mocked(getCurrentUser).mockReturnValue(lookup.promise);

    await onboardingPage();

    await waitFor(() => expect(getCurrentUser).toHaveBeenCalled());
    expect(screen.queryByTestId("step-team")).toBeNull();
    expect(getState).not.toHaveBeenCalled();

    lookup.resolve({ user: USER });

    expect(await screen.findByTestId("step-team")).toBeDefined();
  });
});
