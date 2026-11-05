import { act, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
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
  // Set to fail only the next completion check with this HTTP status.
  checkFailsOnceWith: null as number | null,
  // Set to hold only the next onboarding state read.
  stateHoldOnce: null as Promise<void> | null,
  // Set to fail only the next onboarding state read: an HTTP status, or a
  // request cut off on the way.
  stateFailsOnceWith: null as number | "abort" | null,
}));

vi.mock("@/app/api/__generated__/endpoints/onboarding/onboarding", () => ({
  getV1CheckIfOnboardingIsCompleted: async () => {
    backend.completionChecks += 1;
    // Answered for whoever was signed in when the request went out.
    const isCompleted = backend.isCompleted;
    const hold = backend.holdOnce ?? backend.heldCheck;
    backend.holdOnce = null;
    if (hold) await hold;
    const failure = backend.checkFailsOnceWith;
    backend.checkFailsOnceWith = null;
    if (failure) {
      const { ApiError } = await import("@/lib/autogpt-server-api");
      throw new ApiError("Completion check failed", failure, {});
    }
    return { status: 200, data: { is_completed: isCompleted } };
  },
  getV1OnboardingState: async () => {
    backend.stateReads += 1;
    const completedSteps = [...backend.completedSteps];
    const hold = backend.stateHoldOnce;
    backend.stateHoldOnce = null;
    if (hold) await hold;
    const failure = backend.stateFailsOnceWith;
    backend.stateFailsOnceWith = null;
    if (failure === "abort") {
      throw new DOMException("The request was aborted", "AbortError");
    }
    if (failure) {
      const { ApiError } = await import("@/lib/autogpt-server-api");
      throw new ApiError("State read failed", failure, {});
    }
    return {
      status: 200,
      data: { completedSteps },
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
    vi.useRealTimers();
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
    backend.checkFailsOnceWith = null;
    backend.stateHoldOnce = null;
    backend.stateFailsOnceWith = null;
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

  test("another account signed in straight over the last one gets its own answer", async () => {
    mockPathname = "/marketplace";
    backend.isCompleted = true;
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("true"));

    backend.isCompleted = false;
    mockUserID = "user-b";
    view.rerender(tree());
    expect(completion()).toBe("null");

    await waitFor(() => expect(completion()).toBe("false"));
    expect(backend.completionChecks).toBe(2);
    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
  });

  test("an unfinished user who opens an account link while the check is out stays on it", async () => {
    mockPathname = "/marketplace";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    mockPathname = "/link/abc";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() => expect(completion()).toBe("false"));
    await act(async () => {});
    expect(routerReplace).not.toHaveBeenCalled();
    expect(backend.completionChecks).toBe(1);
  });

  test("an unfinished user who leaves the wizard while its check is out is sent back to it", async () => {
    mockPathname = "/onboarding";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    mockPathname = "/home";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
    expect(backend.completionChecks).toBe(1);
    expect(completion()).toBe("false");
  });

  test.each([
    { finished: true, expected: "true" },
    { finished: false, expected: "false" },
  ])(
    "a user who left the wizard is decided by a fresh check when the state read is cut off (finished: $finished)",
    async ({ finished, expected }) => {
      mockPathname = "/onboarding";
      backend.holdOnce = new Promise<void>((resolve) => {
        releaseHeld = resolve;
      });
      const view = render(tree());
      await waitFor(() => expect(backend.completionChecks).toBe(1));

      // The held check read "not finished" before the wizard was left.
      backend.isCompleted = finished;
      backend.completedSteps = finished ? ["ONBOARDING_COMPLETE"] : [];
      backend.stateFailsOnceWith = "abort";
      mockPathname = "/home";
      view.rerender(tree());
      await act(async () => releaseHeld());

      await waitFor(() => expect(completion()).toBe(expected));
      expect(backend.completionChecks).toBe(2);
      if (finished) {
        expect(routerReplace).not.toHaveBeenCalledWith("/onboarding");
      } else {
        await waitFor(() =>
          expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
        );
      }
    },
  );

  test("a user who finished the wizard is not sent back into it when the state read after it fails", async () => {
    mockPathname = "/onboarding";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    backend.stateFailsOnceWith = 403;
    mockPathname = "/home";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() => expect(backend.stateReads).toBe(1));
    await act(async () => {});
    expect(routerReplace).not.toHaveBeenCalled();
    expect(completion()).not.toBe("false");
  });

  test("an unfinished user opening the app root is sent to the wizard on the first answer, though the root has already moved them to Home", async () => {
    mockPathname = "/";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    mockPathname = "/home";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() =>
      expect(routerReplace).toHaveBeenCalledWith("/onboarding"),
    );
    expect(backend.completionChecks).toBe(1);
  });

  test("a check that answers once the user is on a sign-in page with its own next leaves the routing to that page", async () => {
    mockPathname = "/marketplace";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    window.history.replaceState(null, "", "/login?next=%2Flibrary");
    mockPathname = "/login";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() => expect(completion()).toBe("false"));
    await act(async () => {});
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test.each(["/reset-password", "/logout", "/tour/chat"])(
    "a finished user whose answer arrives on %s still gets their onboarding state",
    async (exempt) => {
      mockPathname = "/settings/account";
      backend.isCompleted = true;
      backend.completedSteps = ["ONBOARDING_COMPLETE"];
      backend.holdOnce = new Promise<void>((resolve) => {
        releaseHeld = resolve;
      });
      const view = render(tree());
      await waitFor(() => expect(backend.completionChecks).toBe(1));

      mockPathname = exempt;
      view.rerender(tree());
      await act(async () => releaseHeld());

      await waitFor(() => expect(backend.stateReads).toBe(1));
      expect(completion()).toBe("true");
      expect(routerReplace).not.toHaveBeenCalled();
    },
  );

  test("a user who finishes the wizard while its earlier check is out is not sent back into it", async () => {
    mockPathname = "/onboarding";
    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(1));

    // The check read "not finished"; then the wizard posted its last step
    // and moved the user to Home.
    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    mockPathname = "/home";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() => expect(completion()).toBe("true"));
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding");
  });

  test("a finished user who opens the wizard while their onboarding state loads is sent Home", async () => {
    mockPathname = "/home";
    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    backend.stateHoldOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    const view = render(tree());
    await waitFor(() => expect(backend.stateReads).toBe(1));

    mockPathname = "/onboarding";
    view.rerender(tree());
    await act(async () => releaseHeld());

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/home"));
    expect(backend.completionChecks).toBe(1);
  });

  test("an admin impersonating an unfinished user is never sent into the target's wizard, and sees it finish", async () => {
    ImpersonationState.set("target-user", "user-a");
    mockPathname = "/admin/users";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));

    mockPathname = "/admin/users/target-user";
    view.rerender(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(2));

    // The target finishes onboarding; the admin's next page picks it up.
    backend.isCompleted = true;
    mockPathname = "/home";
    view.rerender(tree());
    await waitFor(() => expect(completion()).toBe("true"));
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding");
  });

  test("an admin impersonating an unfinished user who lands on a sign-in page goes Home, not into the target's wizard", async () => {
    ImpersonationState.set("target-user", "user-a");
    window.history.replaceState(null, "", "/login");
    mockPathname = "/login";
    render(tree());

    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/home"));
    expect(routerReplace).not.toHaveBeenCalledWith("/onboarding");
  });

  test("an older check answering last can't undo a newer answer or send a finished user back", async () => {
    mockPathname = "/onboarding";
    const view = render(tree());
    await waitFor(() => expect(completion()).toBe("false"));

    backend.holdOnce = new Promise<void>((resolve) => {
      releaseHeld = resolve;
    });
    mockPathname = "/home";
    view.rerender(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(2));

    mockPathname = "/onboarding";
    view.rerender(tree());
    await waitFor(() => expect(backend.completionChecks).toBe(3));

    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    mockPathname = "/home";
    view.rerender(tree());
    await waitFor(() => expect(completion()).toBe("true"));
    routerReplace.mockClear();

    await act(async () => releaseHeld());
    await act(async () => {});

    expect(completion()).toBe("true");
    expect(routerReplace).not.toHaveBeenCalled();
  });

  test("a finished user sent Home from a sign-in page gets their onboarding state on arrival", async () => {
    backend.isCompleted = true;
    backend.completedSteps = ["ONBOARDING_COMPLETE"];
    window.history.replaceState(null, "", "/login");
    mockPathname = "/login";
    const view = render(tree());
    await waitFor(() => expect(routerReplace).toHaveBeenCalledWith("/home"));
    expect(backend.stateReads).toBe(0);

    window.history.replaceState(null, "", "/home");
    mockPathname = "/home";
    view.rerender(tree());

    await waitFor(() => expect(backend.stateReads).toBe(1));
    expect(routerReplace).toHaveBeenCalledTimes(1);
  });

  describe("when the check fails on a sign-in page", () => {
    beforeEach(() => {
      vi.useFakeTimers({
        toFake: ["setTimeout", "clearTimeout"],
        shouldAdvanceTime: true,
      });
      backend.checkFailsOnceWith = 403;
      window.history.replaceState(
        null,
        "",
        "/login?next=%2Fhome%3Futm_source%3Dad",
      );
      mockPathname = "/login";
    });

    async function failedCheckThenWait(ms: number) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(ms);
      });
    }

    test("the user is taken off its loader to Home, keeping tracking, and Home's check takes over", async () => {
      const view = render(tree());
      await waitFor(() => expect(backend.completionChecks).toBe(1));

      await failedCheckThenWait(2_000);
      expect(routerReplace).not.toHaveBeenCalled();

      await failedCheckThenWait(1_500);
      expect(routerReplace).toHaveBeenCalledWith("/home?utm_source=ad");
      expect(completion()).toBe("null");

      window.history.replaceState(null, "", "/home?utm_source=ad");
      mockPathname = "/home";
      view.rerender(tree());
      await waitFor(() =>
        expect(routerReplace).toHaveBeenLastCalledWith("/onboarding"),
      );
      expect(backend.completionChecks).toBe(2);
    });

    test("a dead session goes back to the sign-in form, not Home", async () => {
      backend.checkFailsOnceWith = 401;
      const validateSession = vi.fn(async () => {
        mockUserID = null;
        return false;
      });
      const original = useAuthStore.getState().validateSession;
      useAuthStore.setState({ validateSession });
      try {
        const view = render(tree());
        await waitFor(() => expect(validateSession).toHaveBeenCalledTimes(1));
        view.rerender(tree());
        await failedCheckThenWait(3_500);

        expect(routerReplace).not.toHaveBeenCalled();
      } finally {
        useAuthStore.setState({ validateSession: original });
      }
    });

    test("a session the auth store still accepts goes Home after the wait", async () => {
      backend.checkFailsOnceWith = 401;
      const validateSession = vi.fn(async () => true);
      const original = useAuthStore.getState().validateSession;
      useAuthStore.setState({ validateSession });
      try {
        render(tree());
        await waitFor(() => expect(validateSession).toHaveBeenCalledTimes(1));
        await failedCheckThenWait(3_500);

        expect(routerReplace).toHaveBeenCalledWith("/home?utm_source=ad");
      } finally {
        useAuthStore.setState({ validateSession: original });
      }
    });

    test("nothing happens if the user signed out during the wait", async () => {
      const view = render(tree());
      await waitFor(() => expect(backend.completionChecks).toBe(1));
      await failedCheckThenWait(0);

      mockUserID = null;
      view.rerender(tree());
      await failedCheckThenWait(3_500);

      expect(routerReplace).not.toHaveBeenCalled();
    });

    test("nothing happens if the sign-in page gained its own next during the wait", async () => {
      render(tree());
      await waitFor(() => expect(backend.completionChecks).toBe(1));
      await failedCheckThenWait(0);

      window.history.replaceState(null, "", "/login?next=%2Flibrary");
      await failedCheckThenWait(3_500);

      expect(routerReplace).not.toHaveBeenCalled();
    });
  });
});
