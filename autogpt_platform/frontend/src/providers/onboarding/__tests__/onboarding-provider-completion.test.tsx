import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { ImpersonationState } from "@/lib/impersonation";
import type { WebSocketNotification } from "@/lib/autogpt-server-api";
import OnboardingProvider, { useOnboarding } from "../onboarding-provider";

// The paywall and the out-of-credits prompt wait for
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
  // Set to hold every completion answer until the test releases it.
  heldCheck: null as Promise<void> | null,
  // Set to hold only the next completion answer.
  holdOnce: null as Promise<void> | null,
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: async () => {
    backend.completionChecks += 1;
    // Answered for whoever was signed in when the request went out.
    const isCompleted = backend.isCompleted;
    const hold = backend.holdOnce ?? backend.heldCheck;
    backend.holdOnce = null;
    if (hold) await hold;
    return { status: 200, data: { is_completed: isCompleted } };
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

let releaseHeld = () => {};

describe("OnboardingProvider isOnboardingComplete", () => {
  afterEach(() => {
    releaseHeld();
    ImpersonationState.clear();
  });

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
    backend.holdOnce = null;
    releaseHeld = () => {};
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

  test("another account signed in on the same tab gets its own answer, never the last one's", async () => {
    mockPathname = "/marketplace";
    backend.isCompleted = true;
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("true"));

    // Signed out and back in as someone else without a reload (e.g. from
    // another tab while this one sat on a public page).
    mockUserID = null;
    view.rerender(tree());
    expect(completion()).toBe("null");
    backend.isCompleted = false;
    mockUserID = "user-b";
    view.rerender(tree());

    await waitFor(() => expect(completion()).toBe("false"));
    expect(backend.completionChecks).toBe(2);
    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
  });

  test("finishing the wizard is picked up on the next page even without the websocket notice", async () => {
    mockPathname = "/onboarding";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));

    // The wizard posted ONBOARDING_COMPLETE and sends the user to /home.
    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    mockPathname = "/home";
    view.rerender(tree());

    await waitFor(() => expect(completion()).toBe("true"));
    expect(backend.completionChecks).toBe(2);
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test.each(["/marketplace", "/home", "/library"])(
    "an unfinished user who leaves the wizard for %s is sent back to it",
    async (path) => {
      mockPathname = "/onboarding";
      const view = render(tree());
      await waitFor(() => expect(completion()).toBe("false"));
      expect(routerReplace).not.toHaveBeenCalled();

      mockPathname = path;
      view.rerender(tree());

      await waitFor(() =>
        expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
      );
      expect(completion()).toBe("false");
    },
  );

  test("signing out from the wizard is not bounced back into it", async () => {
    mockPathname = "/onboarding";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));
    const checksInWizard = backend.completionChecks;

    mockPathname = "/logout";
    view.rerender(tree());
    await act(async () => {});

    expect(backend.completionChecks).toBe(checksInWizard);
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
  test("a late answer for the account that signed out changes nothing for the new one", async () => {
    mockPathname = "/marketplace";
    backend.isCompleted = true;
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    // Account A's check is still out when B signs in on this tab.
    backend.isCompleted = false;
    mockUserID = null;
    view.rerender(tree());
    mockUserID = "user-b";
    view.rerender(tree());
    await waitFor(() => expect(completion()).toBe("false"));
    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
    routerReplace.mockClear();

    await act(async () => releaseHeld());
    await act(async () => {});

    expect(completion()).toBe("false");
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("an admin impersonating an unfinished user is not bounced to the wizard on every page", async () => {
    ImpersonationState.set("target-user", "user-a");
    mockPathname = "/admin/users";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));
    const checks = backend.completionChecks;
    const redirects = routerReplace.mock.calls.length;

    mockPathname = "/admin/users/target-user";
    view.rerender(tree());
    await act(async () => {});

    expect(backend.completionChecks).toBe(checks);
    expect(routerReplace.mock.calls.length).toBe(redirects);
  });
});
