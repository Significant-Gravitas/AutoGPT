import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// The store imports a chain of modules (server actions, auth helpers,
// react-query). Stub them so we can exercise `setCurrentRequestContext`
// in isolation, without pulling in next/headers or the Better Auth server
// instance at test time.
vi.mock("../../actions", () => ({
  serverLogout: vi.fn(),
}));
vi.mock("../../helpers", () => ({
  broadcastLogout: vi.fn(),
  navigateWithFullLoad: vi.fn(),
  setWebSocketDisconnectIntent: vi.fn(),
  setupSessionEventListeners: vi.fn(() => ({ cleanup: vi.fn() })),
}));
vi.mock("../../deployment-reload", () => ({
  reloadForNewDeployment: vi.fn(() => true),
}));
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear: vi.fn() }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));
vi.mock("@/services/push-notifications/helpers", () => ({
  teardownPushSubscriptionBeforeLogout: vi.fn(),
}));
const { mockClearAllCompletedSessions } = vi.hoisted(() => ({
  mockClearAllCompletedSessions: vi.fn(),
}));
vi.mock("@/app/(platform)/copilot/store", () => ({
  useCopilotUIStore: {
    getState: () => ({
      clearAllCompletedSessions: mockClearAllCompletedSessions,
    }),
  },
}));
vi.mock("../helpers", () => ({
  fetchUser: vi.fn(),
  handleStorageEvent: vi.fn(),
  refreshSession: vi.fn(),
  validateSession: vi.fn(),
}));

import { IMPERSONATION_STORAGE_KEY } from "@/lib/constants";
import { ImpersonationState } from "@/lib/impersonation";
import { teardownPushSubscriptionBeforeLogout } from "@/services/push-notifications/helpers";
import { serverLogout } from "../../actions";
import { reloadForNewDeployment } from "../../deployment-reload";
import {
  navigateWithFullLoad,
  setupSessionEventListeners,
} from "../../helpers";
import type { User } from "../../types";
import {
  fetchUser,
  handleStorageEvent,
  refreshSession as refreshSessionHelper,
  validateSession as validateSessionHelper,
} from "../helpers";
import { useAuthStore } from "../useAuthStore";

function makeRouter() {
  return {
    push: vi.fn(),
    replace: vi.fn(),
    refresh: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    prefetch: vi.fn(),
  } as unknown as AppRouterInstance;
}

function makeApi() {
  return {} as BackendAPI;
}

function snapshot() {
  const { routerRef, apiRef, currentPath } = useAuthStore.getState();
  return { routerRef, apiRef, currentPath };
}

describe("useAuthStore.setCurrentRequestContext", () => {
  afterEach(() => {
    useAuthStore.setState({
      routerRef: null,
      apiRef: null,
      currentPath: "",
    });
  });

  it("writes refs and path to the store on first call", () => {
    const router = makeRouter();
    const api = makeApi();

    useAuthStore
      .getState()
      .setCurrentRequestContext({ router, api, path: "/library" });

    expect(snapshot()).toEqual({
      routerRef: router,
      apiRef: api,
      currentPath: "/library",
    });
  });

  it("is a no-op when refs and path are identical to current state", () => {
    const router = makeRouter();
    const api = makeApi();
    const { setCurrentRequestContext } = useAuthStore.getState();

    setCurrentRequestContext({ router, api, path: "/library" });
    const setSpy = vi.spyOn(useAuthStore, "setState");

    setCurrentRequestContext({ router, api, path: "/library" });

    expect(setSpy).not.toHaveBeenCalled();
    setSpy.mockRestore();
  });

  it.each([
    {
      label: "router changes",
      next: { router: makeRouter() },
    },
    {
      label: "api changes",
      next: { api: makeApi() },
    },
    {
      label: "path changes",
      next: { path: "/build" },
    },
  ])("writes when $label", ({ next }) => {
    const baseRouter = makeRouter();
    const baseApi = makeApi();
    const basePath = "/library";

    const { setCurrentRequestContext } = useAuthStore.getState();
    setCurrentRequestContext({
      router: baseRouter,
      api: baseApi,
      path: basePath,
    });

    setCurrentRequestContext({
      router: next.router ?? baseRouter,
      api: next.api ?? baseApi,
      path: next.path ?? basePath,
    });

    const after = snapshot();
    expect(after.routerRef).toBe(next.router ?? baseRouter);
    expect(after.apiRef).toBe(next.api ?? baseApi);
    expect(after.currentPath).toBe(next.path ?? basePath);
  });
});

describe("useAuthStore.validateSession", () => {
  afterEach(() => {
    useAuthStore.setState({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
      isValidating: false,
      lastValidation: 0,
    });
    useAuthStore.getState().cleanup();
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("clears the user and redirects when the server says the session is invalid", async () => {
    const router = makeRouter();
    vi.mocked(validateSessionHelper).mockResolvedValue({
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Fbuild",
      shouldUpdateUser: false,
    });
    useAuthStore.setState({
      user: { id: "user-1" } as User,
      hasLoadedUser: true,
    });

    const stillValid = await useAuthStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(stillValid).toBe(false);
    expect(useAuthStore.getState().user).toBeNull();
    expect(router.push).toHaveBeenCalledWith("/login?next=%2Fbuild");
  });

  it("keeps the user and does not redirect when the session is valid", async () => {
    const router = makeRouter();
    const user = { id: "user-1" } as User;
    vi.mocked(validateSessionHelper).mockResolvedValue({
      user,
      isValid: true,
      shouldUpdateUser: false,
    });
    useAuthStore.setState({ user, hasLoadedUser: true });

    const stillValid = await useAuthStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(stillValid).toBe(true);
    expect(useAuthStore.getState().user).toBe(user);
    expect(router.push).not.toHaveBeenCalled();
  });

  it("keeps the user signed in through a network failure and retries", async () => {
    vi.useFakeTimers();
    const router = makeRouter();
    const user = { id: "user-1" } as User;
    vi.mocked(validateSessionHelper)
      .mockResolvedValueOnce({
        isValid: true,
        shouldUpdateUser: false,
        unavailable: "network",
      })
      .mockResolvedValueOnce({
        user: null,
        isValid: false,
        redirectPath: "/login?next=%2Fbuild",
        shouldUpdateUser: true,
      });
    useAuthStore.setState({ user, hasLoadedUser: true });

    const stillValid = await useAuthStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(stillValid).toBe(true);
    expect(useAuthStore.getState().user).toBe(user);
    expect(router.push).not.toHaveBeenCalled();

    useAuthStore.setState({ routerRef: router, currentPath: "/build" });
    await vi.runOnlyPendingTimersAsync();

    expect(validateSessionHelper).toHaveBeenCalledTimes(2);
    expect(useAuthStore.getState().user).toBeNull();
    expect(router.push).toHaveBeenCalledWith("/login?next=%2Fbuild");
  });

  it("reloads onto the new deployment instead of logging out on deploy skew", async () => {
    const router = makeRouter();
    const user = { id: "user-1" } as User;
    vi.mocked(validateSessionHelper).mockResolvedValue({
      isValid: true,
      shouldUpdateUser: false,
      unavailable: "deployment-skew",
    });
    useAuthStore.setState({ user, hasLoadedUser: true });

    await useAuthStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(reloadForNewDeployment).toHaveBeenCalled();
    expect(useAuthStore.getState().user).toBe(user);
    expect(router.push).not.toHaveBeenCalled();
  });
});

describe("useAuthStore resilience", () => {
  const user = { id: "user-1" } as User;

  afterEach(() => {
    useAuthStore.getState().cleanup();
    useAuthStore.setState({
      user: null,
      hasLoadedUser: false,
      isValidating: false,
      lastValidation: 0,
      routerRef: null,
      apiRef: null,
      currentPath: "",
    });
    vi.clearAllMocks();
    vi.useRealTimers();
  });

  it("stops retrying after the budget is spent and never stacks timers", async () => {
    vi.useFakeTimers();
    const router = makeRouter();
    vi.mocked(validateSessionHelper).mockResolvedValue({
      isValid: true,
      shouldUpdateUser: false,
      unavailable: "network",
    });
    useAuthStore.setState({
      user,
      hasLoadedUser: true,
      routerRef: router,
      currentPath: "/library",
    });

    await useAuthStore.getState().validateSession({ force: true });
    await useAuthStore.getState().validateSession({ force: true });
    expect(vi.getTimerCount()).toBe(1);

    for (let i = 0; i < 10; i++) await vi.runOnlyPendingTimersAsync();

    // The first check, the second forced one, then six retries.
    expect(validateSessionHelper).toHaveBeenCalledTimes(8);
    expect(vi.getTimerCount()).toBe(0);
    expect(useAuthStore.getState().user).toBe(user);
    expect(router.push).not.toHaveBeenCalled();
  });

  it("doesn't start a second check when the network returns mid-check", async () => {
    vi.mocked(fetchUser).mockResolvedValue({
      user,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    await useAuthStore.getState().initialize({
      api: makeApi(),
      router: makeRouter(),
      path: "/library",
    });
    const [, , onOnline] = vi.mocked(setupSessionEventListeners).mock.calls[0];
    let finish!: (
      result: Awaited<ReturnType<typeof validateSessionHelper>>,
    ) => void;
    vi.mocked(validateSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );

    const inFlight = useAuthStore.getState().validateSession({ force: true });
    onOnline?.();
    finish({ user, isValid: true, shouldUpdateUser: false });
    await inFlight;

    expect(validateSessionHelper).toHaveBeenCalledTimes(1);
  });

  it("doesn't spend a retry when its timer fires while a check is already running", async () => {
    vi.useFakeTimers();
    const router = makeRouter();
    useAuthStore.setState({
      user,
      hasLoadedUser: true,
      routerRef: router,
      currentPath: "/library",
    });
    vi.mocked(validateSessionHelper).mockResolvedValueOnce({
      isValid: true,
      shouldUpdateUser: false,
      unavailable: "network",
    });
    await useAuthStore.getState().validateSession({ force: true });
    expect(useAuthStore.getState().retryAttempt).toBe(1);

    let finish!: (
      result: Awaited<ReturnType<typeof validateSessionHelper>>,
    ) => void;
    vi.mocked(validateSessionHelper).mockReturnValueOnce(
      new Promise((resolve) => {
        finish = resolve;
      }),
    );
    const inFlight = useAuthStore
      .getState()
      .validateSession({ force: true, router, path: "/library" });
    await vi.runOnlyPendingTimersAsync();

    expect(validateSessionHelper).toHaveBeenCalledTimes(2);
    expect(useAuthStore.getState().retryAttempt).toBe(0);
    finish({ user, isValid: true, shouldUpdateUser: false });
    await inFlight;
  });

  it("sends the tab to login with a full page load when another tab signs out", async () => {
    vi.mocked(fetchUser).mockResolvedValue({
      user,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    vi.mocked(handleStorageEvent).mockReturnValue({
      shouldLogout: true,
      redirectPath: "/login?next=%2Flibrary",
    });
    await useAuthStore.getState().initialize({
      api: makeApi(),
      router: makeRouter(),
      path: "/library",
    });
    const [, onStorage] = vi.mocked(setupSessionEventListeners).mock.calls[0];

    onStorage(new StorageEvent("storage", { key: "supabase-logout" }));

    expect(useAuthStore.getState().user).toBeNull();
    expect(navigateWithFullLoad).toHaveBeenCalledWith("/login?next=%2Flibrary");
  });

  it("keeps the user when a session refresh can't reach the auth backend", async () => {
    useAuthStore.setState({ user, hasLoadedUser: true });
    vi.mocked(refreshSessionHelper).mockResolvedValue({
      user: null,
      error: "connection reset",
      isUnavailable: true,
    });

    await useAuthStore.getState().refreshSession();

    expect(useAuthStore.getState().user).toBe(user);
  });

  it("clears the user when a session refresh finds no session", async () => {
    useAuthStore.setState({ user, hasLoadedUser: true });
    vi.mocked(refreshSessionHelper).mockResolvedValue({
      user: null,
      error: "No active session",
    });

    await useAuthStore.getState().refreshSession();

    expect(useAuthStore.getState().user).toBeNull();
  });
});

describe("useAuthStore.logOut", () => {
  afterEach(async () => {
    useAuthStore.setState({ user: null, hasLoadedUser: false });
    await vi.dynamicImportSettled();
    vi.clearAllMocks();
  });

  it("clears admin impersonation so the next user doesn't act as someone else", async () => {
    ImpersonationState.set("impersonated-user", "admin-1");
    expect(ImpersonationState.read()?.target).toBe("impersonated-user");

    await useAuthStore.getState().logOut();

    expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
    expect(document.cookie).not.toContain(`${IMPERSONATION_STORAGE_KEY}=`);
    expect(ImpersonationState.read()).toBeNull();
  });

  it("waits for this browser's push subscription to go before the server signs out", async () => {
    let finishTeardown: () => void = () => {};
    vi.mocked(teardownPushSubscriptionBeforeLogout).mockReturnValue(
      new Promise<void>((resolve) => (finishTeardown = resolve)),
    );
    useAuthStore.setState({ user: { id: "user-1" } as User });

    const loggingOut = useAuthStore.getState().logOut();
    await Promise.resolve();

    expect(teardownPushSubscriptionBeforeLogout).toHaveBeenCalledOnce();
    expect(serverLogout).not.toHaveBeenCalled();

    finishTeardown();
    await loggingOut;

    expect(serverLogout).toHaveBeenCalledOnce();
  });
});

describe("useAuthStore onboarding handoff", () => {
  // The wizard's handoff to Home lives in this tab's sessionStorage, which
  // outlives a sign-out: the next account to sign in here must not inherit
  // "just finished onboarding" (and get the first-run walkthrough).
  const WELCOME = "autogpt:onboarding-welcome-pending";
  const INTRO = "autogpt:onboarding-intro-path";

  beforeEach(() => {
    window.sessionStorage.setItem(WELCOME, "1");
    window.sessionStorage.setItem(INTRO, "A");
  });

  afterEach(async () => {
    window.sessionStorage.clear();
    useAuthStore.setState({ user: null, hasLoadedUser: false });
    await vi.dynamicImportSettled();
    vi.clearAllMocks();
  });

  it("is cleared when the user signs out", async () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });

    await useAuthStore.getState().logOut();

    expect(window.sessionStorage.getItem(WELCOME)).toBeNull();
    expect(window.sessionStorage.getItem(INTRO)).toBeNull();
  });

  it("is cleared when another account takes over the tab", () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: { id: "user-2" } as User });

    expect(window.sessionStorage.getItem(WELCOME)).toBeNull();
  });

  it("is cleared when the next account signs in after a session ended some other way", () => {
    // A sign-out in another tab or a failed session check drops the user
    // without logOut(); the next sign-in still must not inherit the flags.
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: null });
    useAuthStore.setState({ user: { id: "user-2" } as User });

    expect(window.sessionStorage.getItem(WELCOME)).toBeNull();
    expect(window.sessionStorage.getItem(INTRO)).toBeNull();
  });

  it("survives the same user signing back in after a dropped session", () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: null });
    useAuthStore.setState({ user: { id: "user-1" } as User });

    expect(window.sessionStorage.getItem(WELCOME)).toBe("1");
  });

  it("survives the same user being refreshed", () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: { id: "user-1" } as User });

    expect(window.sessionStorage.getItem(WELCOME)).toBe("1");
    expect(window.sessionStorage.getItem(INTRO)).toBe("A");
  });
});

describe("useAuthStore account change", () => {
  afterEach(async () => {
    useAuthStore.setState({ user: null });
    await vi.dynamicImportSettled();
    vi.clearAllMocks();
  });

  it("clears the finished-chats list when the user signs out", async () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: null });

    await vi.waitFor(() =>
      expect(mockClearAllCompletedSessions).toHaveBeenCalledOnce(),
    );
  });

  it("keeps the finished-chats list when the same user is refreshed", async () => {
    useAuthStore.setState({ user: { id: "user-1" } as User });
    useAuthStore.setState({ user: { id: "user-1" } as User });
    await vi.dynamicImportSettled();

    expect(mockClearAllCompletedSessions).not.toHaveBeenCalled();
  });
});
