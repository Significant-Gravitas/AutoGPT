import {
  getGetV1OnboardingStateMockHandler200,
  getGetV1OnboardingStateResponseMock,
  getPatchV1UpdateOnboardingStateMockHandler200,
  getPatchV1UpdateOnboardingStateResponseMock,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding.msw";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import type BackendAPI from "@/lib/autogpt-server-api/client";
import { server } from "@/mocks/mock-server";
import {
  createRouterQueue,
  disposeRouterQueue,
} from "@/tests/integrations/next-router-queue";
import { render, screen } from "@/tests/integrations/test-utils";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import OnboardingPage from "../page";
import { useOnboardingWizardStore } from "../store";

// #16965. Signed out, `/` lands on `/login?next=/home`, and the login page's
// sign-in check is still waiting on its first answer when "Sign up" is
// clicked. The check's second question then went out while that navigation
// was running, and Next's router queue dropped it: the check never finished,
// and every later page's check joined it, so after signing up onboarding sat
// on "Loading your setup…" for good.
//
// Only the server actions are faked, and they go through Next's own router
// queue the way the browser sends them.

const auth = vi.hoisted(() => ({
  queue: null as null | {
    callServer: <T>(run: () => Promise<T>) => Promise<T>;
  },
  signedIn: null as User | null,
  firstAnswer: Promise.resolve(),
}));

function serverAction<T>(run: () => Promise<T>) {
  if (!auth.queue) throw new Error("No router queue");
  return auth.queue.callServer(run);
}

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: () =>
    serverAction(async () => {
      await auth.firstAnswer;
      return { user: auth.signedIn };
    }),
  validateSession: () =>
    serverAction(async () =>
      auth.signedIn
        ? { user: auth.signedIn, isValid: true }
        : { user: null, isValid: false },
    ),
  refreshSession: () => serverAction(async () => ({ user: auth.signedIn })),
  serverLogout: vi.fn(),
  getWebSocketToken: () => Promise.resolve({ token: null }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));
vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));
const router = vi.hoisted(() => ({
  replace: vi.fn(),
  push: vi.fn(),
  refresh: vi.fn(),
  prefetch: vi.fn(),
  back: vi.fn(),
  forward: vi.fn(),
}));
vi.mock("next/navigation", () => ({
  useRouter: () => router,
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/onboarding",
}));

const NEW_USER: User = {
  id: "new-user",
  email: "new@example.com",
  user_metadata: {},
};

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  useOnboardingWizardStore.getState().reset();
  auth.signedIn = null;
  server.use(
    getGetV1OnboardingStateMockHandler200(
      getGetV1OnboardingStateResponseMock({
        userId: NEW_USER.id,
        completedSteps: [],
        wizardProgress: null,
        wizardRevision: 0,
      }),
    ),
    getPatchV1UpdateOnboardingStateMockHandler200(() =>
      getPatchV1UpdateOnboardingStateResponseMock({
        userId: NEW_USER.id,
        wizardRevision: 1,
      }),
    ),
  );
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  useAuthStore.setState({
    user: null,
    hasLoadedUser: false,
    isUserLoading: true,
    isValidating: false,
    initializationPromise: null,
    lastValidation: 0,
    routerRef: null,
    apiRef: null,
    currentPath: "",
  });
  auth.queue = null;
  disposeRouterQueue();
});

function afterAnswersArrive() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("signing up right after landing on the login page", () => {
  test("onboarding loads its first step once the account exists", async () => {
    const queue = createRouterQueue();
    auth.queue = queue;
    let answerFirstCheck = () => {};
    auth.firstAnswer = new Promise((resolve) => {
      answerFirstCheck = resolve;
    });

    void useAuthStore.getState().initialize({
      api: {} as BackendAPI,
      router: router as unknown as AppRouterInstance,
      path: "/login?next=%2Fhome",
    });
    await afterAnswersArrive();
    const toSignup = queue.navigate();
    answerFirstCheck();
    await afterAnswersArrive();
    toSignup.arrive();
    await afterAnswersArrive();

    auth.signedIn = NEW_USER;
    render(<OnboardingPage />);

    expect(await screen.findByTestId("intro-stage")).toBeDefined();
  });
});
