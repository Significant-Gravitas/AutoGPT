import { act, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, test, vi } from "vitest";
import type { WebSocketNotification } from "@/lib/autogpt-server-api";
import OnboardingProvider, { useOnboarding } from "../onboarding-provider";

// SECRT-3384: the paywall and the out-of-credits prompt wait for
// `isOnboardingComplete === true`, so it must be `null` until known, `false`
// for an unfinished user, and reach `true` once the wizard is done.

const { routerReplace, router, api, notificationHandlers } = vi.hoisted(() => {
  const replace = vi.fn();
  const handlers: Array<(n: WebSocketNotification) => void> = [];
  return {
    routerReplace: replace,
    router: { replace, push: () => {}, refresh: () => {} },
    notificationHandlers: handlers,
    api: {
      onWebSocketMessage: (
        _type: string,
        handler: (n: WebSocketNotification) => void,
      ) => {
        handlers.push(handler);
        return () => {};
      },
      connectWebSocket: () => {},
    },
  };
});

let mockPathname = "/home";
vi.mock("next/navigation", () => ({
  useRouter: () => router,
  usePathname: () => mockPathname,
  useSearchParams: () => new URLSearchParams(window.location.search),
}));

let mockUserID: string | null = "user-a";
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    isLoggedIn: mockUserID !== null,
    isUserLoading: false,
    user: mockUserID ? { id: mockUserID } : null,
  }),
}));

vi.mock("@/lib/autogpt-server-api/context", () => ({
  useBackendAPI: () => api,
}));

vi.mock("@/hooks/useOnboardingTimezoneDetection", () => ({
  useOnboardingTimezoneDetection: () => undefined,
}));

vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: () => {} }),
}));

const backend = vi.hoisted(() => ({
  isCompleted: false,
  completedSteps: [] as string[],
  completionChecks: 0,
  stateReads: 0,
  // Set to hold the next completion answer until the test releases it.
  heldCheck: null as Promise<void> | null,
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: async () => {
    backend.completionChecks += 1;
    if (backend.heldCheck) await backend.heldCheck;
    return { status: 200, data: { is_completed: backend.isCompleted } };
  },
  getV1OnboardingState: async () => {
    backend.stateReads += 1;
    return {
      status: 200,
      data: { completedSteps: [...backend.completedSteps] },
    };
  },
  patchV1UpdateOnboardingState: () => Promise.resolve({ status: 200 }),
  postV1CompleteOnboardingStep: () => Promise.resolve({ status: 200 }),
}));

function Probe() {
  const { isOnboardingComplete } = useOnboarding();
  return <span data-testid="complete">{String(isOnboardingComplete)}</span>;
}

function tree() {
  return (
    <OnboardingProvider>
      <Probe />
    </OnboardingProvider>
  );
}

function completion() {
  return screen.getByTestId("complete").textContent;
}

describe("OnboardingProvider isOnboardingComplete", () => {
  beforeEach(() => {
    routerReplace.mockClear();
    notificationHandlers.length = 0;
    window.history.replaceState(null, "", "/");
    mockPathname = "/home";
    mockUserID = "user-a";
    backend.isCompleted = false;
    backend.completedSteps = [];
    backend.completionChecks = 0;
    backend.stateReads = 0;
    backend.heldCheck = null;
  });

  test("is null until the check answers, then false for an unfinished user on Home", async () => {
    let release = () => {};
    backend.heldCheck = new Promise<void>((resolve) => {
      release = resolve;
    });

    render(tree());

    await waitFor(() => expect(backend.completionChecks).toBe(1));
    expect(completion()).toBe("null");

    await act(async () => release());

    await waitFor(() => expect(completion()).toBe("false"));
    expect(routerReplace).toHaveBeenCalledWith("/onboarding");
  });

  test("is true for a user who has finished onboarding", async () => {
    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];

    render(tree());

    await waitFor(() => expect(completion()).toBe("true"));
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("is null while signed out, without asking the backend", async () => {
    mockUserID = null;

    render(tree());
    await act(async () => {});

    expect(completion()).toBe("null");
    expect(backend.completionChecks).toBe(0);
  });

  test("a different account in the same tab does not inherit the answer", async () => {
    backend.isCompleted = true;
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("true"));

    mockUserID = "user-b";
    view.rerender(tree());

    expect(completion()).toBe("null");
  });

  test("leaving the wizard after finishing it re-reads onboarding, so the prompts can show", async () => {
    mockPathname = "/onboarding";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));
    const readsInWizard = backend.stateReads;

    // The wizard posted ONBOARDING_COMPLETE, but the websocket notice never
    // arrived; it now sends the user to /home.
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    mockPathname = "/home";
    view.rerender(tree());

    await waitFor(() => expect(completion()).toBe("true"));
    expect(backend.stateReads).toBe(readsInWizard + 1);
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("leaving the wizard unfinished (e.g. to log out) stays false and does not redirect", async () => {
    mockPathname = "/onboarding";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));
    const readsInWizard = backend.stateReads;

    mockPathname = "/logout";
    view.rerender(tree());

    await waitFor(() => expect(backend.stateReads).toBe(readsInWizard + 1));
    await act(async () => {});
    expect(completion()).toBe("false");
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("an onboarding notification that completes onboarding flips it to true", async () => {
    mockPathname = "/onboarding";
    render(tree());
    await waitFor(() => expect(completion()).toBe("false"));

    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    await act(async () => {
      notificationHandlers[notificationHandlers.length - 1]({
        type: "onboarding",
      } as WebSocketNotification);
    });

    await waitFor(() => expect(completion()).toBe("true"));
  });
});
