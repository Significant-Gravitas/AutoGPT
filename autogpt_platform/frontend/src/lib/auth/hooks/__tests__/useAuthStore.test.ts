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
  setWebSocketDisconnectIntent: vi.fn(),
  setupSessionEventListeners: vi.fn(() => ({ cleanup: vi.fn() })),
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

import { teardownPushSubscriptionBeforeLogout } from "@/services/push-notifications/helpers";
import { serverLogout } from "../../actions";
import type { User } from "../../types";
import { validateSession as validateSessionHelper } from "../helpers";
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
    vi.clearAllMocks();
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
});

describe("useAuthStore.logOut", () => {
  afterEach(async () => {
    useAuthStore.setState({ user: null, hasLoadedUser: false });
    await vi.dynamicImportSettled();
    vi.clearAllMocks();
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
