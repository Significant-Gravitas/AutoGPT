import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

const resetQueryCache = vi.hoisted(() => vi.fn());
const listenerCleanup = vi.hoisted(() => vi.fn());

// The store imports a chain of modules (server actions, auth helpers,
// react-query). Stub them so we can exercise `setCurrentRequestContext`
// in isolation, without pulling in next/headers or the Better Auth server
// instance at test time.
vi.mock("../../actions", () => ({
  serverLogout: vi.fn(async () => ({ success: true })),
}));
vi.mock("../../helpers", () => ({
  broadcastLogout: vi.fn(),
  clearWebSocketDisconnectIntent: vi.fn(),
  setWebSocketDisconnectIntent: vi.fn(),
  setupSessionEventListeners: vi.fn(() => ({ cleanup: listenerCleanup })),
}));
vi.mock("../helpers", () => ({
  fetchUser: vi.fn(),
  handleStorageEvent: vi.fn(),
  refreshSession: vi.fn(),
  validateSession: vi.fn(),
}));
vi.mock("@/lib/react-query/queryClient", () => ({
  resetQueryClientForIdentityChange: resetQueryCache,
}));

import type { User } from "../../types";
import { serverLogout } from "../../actions";
import {
  broadcastLogout,
  clearWebSocketDisconnectIntent,
  setupSessionEventListeners,
} from "../../helpers";
import {
  fetchUser,
  handleStorageEvent as handleStorageEventHelper,
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
  return { disconnectWebSocket: vi.fn() } as unknown as BackendAPI;
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
      hasCompletedInitialAuthHydration: false,
      isUserLoading: false,
      initializationPromise: null,
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

describe("useAuthStore identity cache isolation", () => {
  afterEach(() => {
    useAuthStore.getState().cleanup();
    useAuthStore.setState({
      user: null,
      hasLoadedUser: false,
      hasCompletedInitialAuthHydration: false,
      isUserLoading: false,
      initializationPromise: null,
      routerRef: null,
      apiRef: null,
      currentPath: "",
      isValidating: false,
      lastValidation: 0,
    });
    vi.clearAllMocks();
  });

  it("preserves hydrated queries when loading the first identity", async () => {
    useAuthStore.setState({
      user: null,
      hasLoadedUser: false,
      hasCompletedInitialAuthHydration: false,
      isUserLoading: true,
      initializationPromise: null,
    });
    vi.mocked(fetchUser).mockResolvedValue({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    resetQueryCache.mockClear();

    await useAuthStore.getState().initialize({
      router: makeRouter(),
      api: makeApi(),
      path: "/library",
    });

    expect(resetQueryCache).not.toHaveBeenCalled();
    expect(clearWebSocketDisconnectIntent).toHaveBeenCalledOnce();
    expect(useAuthStore.getState().hasCompletedInitialAuthHydration).toBe(true);
  });

  it("clears cached queries when an authenticated identity changes", () => {
    useAuthStore.setState({ hasCompletedInitialAuthHydration: true });
    resetQueryCache.mockClear();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
    });
    expect(resetQueryCache).toHaveBeenLastCalledWith("user-a");

    useAuthStore.setState({ user: { id: "user-b" } as User });
    expect(resetQueryCache).toHaveBeenLastCalledWith("user-b");

    useAuthStore.setState({ user: null });
    expect(resetQueryCache).toHaveBeenLastCalledWith(null);
    expect(resetQueryCache).toHaveBeenCalledTimes(3);
  });

  it("refetches when an identity arrives after initial anonymous hydration", () => {
    useAuthStore.setState({
      user: null,
      hasCompletedInitialAuthHydration: true,
    });
    resetQueryCache.mockClear();

    useAuthStore.setState({ user: { id: "user-b" } as User });

    expect(resetQueryCache).toHaveBeenCalledOnce();
    expect(resetQueryCache).toHaveBeenCalledWith("user-b");
  });

  it("does not restore a user when logout wins an initial fetch race", async () => {
    let resolveFetch:
      | ((value: {
          user: User | null;
          hasLoadedUser: boolean;
          isUserLoading: boolean;
        }) => void)
      | undefined;
    vi.mocked(fetchUser).mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    const api = makeApi();

    const initialization = useAuthStore
      .getState()
      .initialize({ api, router: makeRouter(), path: "/logout" });
    await vi.waitFor(() => expect(fetchUser).toHaveBeenCalledOnce());
    await useAuthStore.getState().logOut({ api });

    resolveFetch?.({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    await initialization;

    expect(useAuthStore.getState().user).toBeNull();
    expect(clearWebSocketDisconnectIntent).not.toHaveBeenCalled();
    expect(setupSessionEventListeners).toHaveBeenCalledOnce();
    expect(listenerCleanup).toHaveBeenCalledOnce();
    expect(resetQueryCache).toHaveBeenCalledWith(null);
  });

  it("clears cached queries when logout lands before hydration completes", async () => {
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: false,
    });
    resetQueryCache.mockClear();

    await useAuthStore.getState().logOut({ api: makeApi() });

    expect(resetQueryCache).toHaveBeenCalledOnce();
    expect(resetQueryCache).toHaveBeenCalledWith(null);
  });

  it("keeps a fresh post-logout initialization when an older one finishes", async () => {
    let resolveOldFetch:
      | ((value: {
          user: User | null;
          hasLoadedUser: boolean;
          isUserLoading: boolean;
        }) => void)
      | undefined;
    vi.mocked(fetchUser)
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveOldFetch = resolve;
        }),
      )
      .mockResolvedValueOnce({
        user: { id: "user-b" } as User,
        hasLoadedUser: true,
        isUserLoading: false,
      });
    const api = makeApi();
    const router = makeRouter();

    const oldInitialization = useAuthStore
      .getState()
      .initialize({ api, router, path: "/logout" });
    await vi.waitFor(() => expect(fetchUser).toHaveBeenCalledOnce());
    await useAuthStore.getState().logOut({ api });

    await useAuthStore.getState().initialize({ api, router, path: "/library" });
    resolveOldFetch?.({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    await oldInitialization;

    expect(useAuthStore.getState().user?.id).toBe("user-b");
    expect(useAuthStore.getState().initializationPromise).toBeNull();
    expect(clearWebSocketDisconnectIntent).toHaveBeenCalledOnce();
    expect(resetQueryCache).toHaveBeenCalledWith("user-b");
    expect(setupSessionEventListeners).toHaveBeenCalledTimes(2);
  });

  it("waits for a pending logout before starting a new initialization", async () => {
    let resolveLogout: ((value: { success: boolean }) => void) | undefined;
    vi.mocked(serverLogout).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveLogout = resolve;
      }),
    );
    vi.mocked(fetchUser).mockResolvedValue({
      user: { id: "user-b" } as User,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    const api = makeApi();
    const router = makeRouter();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const logout = useAuthStore.getState().logOut({ api });
    const initialization = useAuthStore
      .getState()
      .initialize({ api, router, path: "/library" });
    await Promise.resolve();

    expect(fetchUser).not.toHaveBeenCalled();
    expect(broadcastLogout).not.toHaveBeenCalled();
    resolveLogout?.({ success: true });
    await logout;
    await initialization;

    expect(fetchUser).toHaveBeenCalledOnce();
    expect(broadcastLogout).toHaveBeenCalledOnce();
    expect(useAuthStore.getState().user?.id).toBe("user-b");
  });

  it("stops loading when a pending logout fails before initialization", async () => {
    let rejectLogout: ((reason?: unknown) => void) | undefined;
    vi.mocked(serverLogout).mockReturnValueOnce(
      new Promise<{ success: boolean }>((_resolve, reject) => {
        rejectLogout = reject;
      }),
    );
    const api = makeApi();

    const logout = useAuthStore.getState().logOut({ api });
    const initialization = useAuthStore.getState().initialize({
      api,
      router: makeRouter(),
      path: "/library",
    });
    await Promise.resolve();

    expect(useAuthStore.getState().isUserLoading).toBe(true);
    const logoutAssertion = expect(logout).rejects.toThrow("logout failed");
    rejectLogout?.(new Error("logout failed"));
    await logoutAssertion;
    await initialization;

    expect(fetchUser).not.toHaveBeenCalled();
    expect(useAuthStore.getState().user).toBeNull();
    expect(useAuthStore.getState().isUserLoading).toBe(false);
    expect(useAuthStore.getState().initializationPromise).toBeNull();
  });

  it("stops loading when refresh supersedes initialization and logout fails", async () => {
    let rejectLogout: ((reason?: unknown) => void) | undefined;
    vi.mocked(serverLogout).mockReturnValueOnce(
      new Promise<{ success: boolean }>((_resolve, reject) => {
        rejectLogout = reject;
      }),
    );
    const api = makeApi();
    const router = makeRouter();

    const logout = useAuthStore.getState().logOut({ api });
    const initialization = useAuthStore
      .getState()
      .initialize({ api, router, path: "/library" });
    const refresh = useAuthStore.getState().refreshSession();
    await Promise.resolve();

    const logoutAssertion = expect(logout).rejects.toThrow("logout failed");
    rejectLogout?.(new Error("logout failed"));
    await logoutAssertion;
    await initialization;
    await refresh;

    expect(fetchUser).not.toHaveBeenCalled();
    expect(refreshSessionHelper).not.toHaveBeenCalled();
    expect(useAuthStore.getState().user).toBeNull();
    expect(useAuthStore.getState().isUserLoading).toBe(false);
  });

  it("keeps listeners when refresh supersedes initialization during logout", async () => {
    let resolveLogout: ((value: { success: boolean }) => void) | undefined;
    vi.mocked(serverLogout).mockReturnValueOnce(
      new Promise((resolve) => {
        resolveLogout = resolve;
      }),
    );
    vi.mocked(refreshSessionHelper).mockResolvedValue({
      user: { id: "user-b" } as User,
    });
    const api = makeApi();
    const router = makeRouter();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const logout = useAuthStore.getState().logOut({ api });
    const initialization = useAuthStore
      .getState()
      .initialize({ api, router, path: "/library" });
    const refresh = useAuthStore.getState().refreshSession();
    await Promise.resolve();

    expect(setupSessionEventListeners).toHaveBeenCalledOnce();
    expect(fetchUser).not.toHaveBeenCalled();
    expect(refreshSessionHelper).not.toHaveBeenCalled();

    resolveLogout?.({ success: true });
    await logout;
    await initialization;
    await refresh;

    expect(fetchUser).not.toHaveBeenCalled();
    expect(refreshSessionHelper).toHaveBeenCalledOnce();
    expect(useAuthStore.getState().user?.id).toBe("user-b");
    expect(useAuthStore.getState().listenersCleanup).toBe(listenerCleanup);
  });

  it("does not restore a user when logout wins fallback validation", async () => {
    let resolveValidation:
      | ((value: {
          user: User | null;
          isValid: boolean;
          shouldUpdateUser: boolean;
        }) => void)
      | undefined;
    vi.mocked(fetchUser).mockResolvedValue({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });
    vi.mocked(validateSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    const api = makeApi();

    const initialization = useAuthStore
      .getState()
      .initialize({ api, router: makeRouter(), path: "/logout" });
    await vi.waitFor(() =>
      expect(validateSessionHelper).toHaveBeenCalledOnce(),
    );
    await useAuthStore.getState().logOut({ api });

    resolveValidation?.({
      user: { id: "user-a" } as User,
      isValid: true,
      shouldUpdateUser: true,
    });
    await initialization;

    expect(useAuthStore.getState().user).toBeNull();
    expect(clearWebSocketDisconnectIntent).not.toHaveBeenCalled();
    expect(setupSessionEventListeners).toHaveBeenCalledOnce();
    expect(listenerCleanup).toHaveBeenCalledOnce();
  });

  it("does not restore a user when logout wins session validation", async () => {
    let resolveValidation:
      | ((value: {
          user: User | null;
          isValid: boolean;
          shouldUpdateUser: boolean;
        }) => void)
      | undefined;
    vi.mocked(validateSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    const api = makeApi();
    const router = makeRouter();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const validation = useAuthStore
      .getState()
      .validateSession({ router, path: "/library", force: true });
    await vi.waitFor(() =>
      expect(validateSessionHelper).toHaveBeenCalledOnce(),
    );
    await useAuthStore.getState().logOut({ api });

    resolveValidation?.({
      user: { id: "user-a" } as User,
      isValid: true,
      shouldUpdateUser: true,
    });

    await expect(validation).resolves.toBe(false);
    expect(useAuthStore.getState().user).toBeNull();
    expect(clearWebSocketDisconnectIntent).not.toHaveBeenCalled();
  });

  it("does not restore a user when logout wins session refresh", async () => {
    let resolveRefresh:
      | ((value: { user?: User | null; error?: string }) => void)
      | undefined;
    vi.mocked(refreshSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveRefresh = resolve;
      }),
    );
    const api = makeApi();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const refresh = useAuthStore.getState().refreshSession();
    await vi.waitFor(() => expect(refreshSessionHelper).toHaveBeenCalledOnce());
    await useAuthStore.getState().logOut({ api });

    resolveRefresh?.({ user: { id: "user-a" } as User });
    await refresh;

    expect(useAuthStore.getState().user).toBeNull();
    expect(clearWebSocketDisconnectIntent).not.toHaveBeenCalled();
  });

  it("rejects an initial fetch after a cross-tab logout broadcast", async () => {
    let resolveFetch:
      | ((value: {
          user: User | null;
          hasLoadedUser: boolean;
          isUserLoading: boolean;
        }) => void)
      | undefined;
    vi.mocked(fetchUser).mockReturnValue(
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
    );
    vi.mocked(handleStorageEventHelper).mockReturnValueOnce({
      shouldLogout: true,
    });

    const initialization = useAuthStore.getState().initialize({
      api: makeApi(),
      router: makeRouter(),
      path: "/library",
    });
    await vi.waitFor(() => expect(fetchUser).toHaveBeenCalledOnce());
    const storageHandler = vi.mocked(setupSessionEventListeners).mock
      .calls[0]?.[1];

    storageHandler?.(new StorageEvent("storage", { key: "auth-logout" }));
    resolveFetch?.({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    await initialization;

    expect(useAuthStore.getState().user).toBeNull();
    expect(clearWebSocketDisconnectIntent).not.toHaveBeenCalled();
    expect(resetQueryCache).toHaveBeenCalledWith(null);
  });

  it("keeps the newest auth operation when an older refresh finishes", async () => {
    let resolveRefresh:
      | ((value: { user?: User | null; error?: string }) => void)
      | undefined;
    vi.mocked(refreshSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveRefresh = resolve;
      }),
    );
    vi.mocked(validateSessionHelper).mockResolvedValue({
      user: { id: "user-b" } as User,
      isValid: true,
      shouldUpdateUser: true,
    });
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const refresh = useAuthStore.getState().refreshSession();
    await vi.waitFor(() => expect(refreshSessionHelper).toHaveBeenCalledOnce());
    await expect(
      useAuthStore.getState().validateSession({
        router: makeRouter(),
        path: "/library",
        force: true,
      }),
    ).resolves.toBe(true);

    resolveRefresh?.({ user: { id: "user-a" } as User });
    await refresh;

    expect(useAuthStore.getState().user?.id).toBe("user-b");
  });

  it("does not let a loaded-user initialize supersede validation", async () => {
    let resolveValidation:
      | ((value: {
          user: User | null;
          isValid: boolean;
          redirectPath?: string;
          shouldUpdateUser: boolean;
        }) => void)
      | undefined;
    vi.mocked(validateSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveValidation = resolve;
      }),
    );
    const router = makeRouter();
    const api = makeApi();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const validation = useAuthStore
      .getState()
      .validateSession({ router, path: "/library", force: true });
    await vi.waitFor(() =>
      expect(validateSessionHelper).toHaveBeenCalledOnce(),
    );
    await useAuthStore.getState().initialize({
      api,
      router,
      path: "/library",
    });

    resolveValidation?.({
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Flibrary",
      shouldUpdateUser: true,
    });

    await expect(validation).resolves.toBe(false);
    expect(useAuthStore.getState().user).toBeNull();
    expect(router.push).toHaveBeenCalledWith("/login?next=%2Flibrary");
  });

  it("does not let a loaded-user initialize supersede refresh", async () => {
    let resolveRefresh:
      | ((value: { user?: User | null; error?: string }) => void)
      | undefined;
    vi.mocked(refreshSessionHelper).mockReturnValue(
      new Promise((resolve) => {
        resolveRefresh = resolve;
      }),
    );
    const router = makeRouter();
    const api = makeApi();
    useAuthStore.setState({
      user: { id: "user-a" } as User,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const refresh = useAuthStore.getState().refreshSession();
    await vi.waitFor(() => expect(refreshSessionHelper).toHaveBeenCalledOnce());
    await useAuthStore.getState().initialize({
      api,
      router,
      path: "/library",
    });

    resolveRefresh?.({ user: { id: "user-b" } as User });
    await refresh;

    expect(useAuthStore.getState().user?.id).toBe("user-b");
  });

  it("keeps the validation flag owned by the newest validation", async () => {
    let resolveFirst:
      | ((value: {
          user: User | null;
          isValid: boolean;
          shouldUpdateUser: boolean;
        }) => void)
      | undefined;
    let resolveSecond:
      | ((value: {
          user: User | null;
          isValid: boolean;
          shouldUpdateUser: boolean;
        }) => void)
      | undefined;
    vi.mocked(validateSessionHelper)
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveFirst = resolve;
        }),
      )
      .mockReturnValueOnce(
        new Promise((resolve) => {
          resolveSecond = resolve;
        }),
      );
    const user = { id: "user-a" } as User;
    const router = makeRouter();
    useAuthStore.setState({
      user,
      hasLoadedUser: true,
      hasCompletedInitialAuthHydration: true,
    });

    const first = useAuthStore
      .getState()
      .validateSession({ router, path: "/library", force: true });
    await vi.waitFor(() =>
      expect(validateSessionHelper).toHaveBeenCalledTimes(1),
    );
    const second = useAuthStore
      .getState()
      .validateSession({ router, path: "/library", force: true });
    await vi.waitFor(() =>
      expect(validateSessionHelper).toHaveBeenCalledTimes(2),
    );

    resolveFirst?.({ user, isValid: true, shouldUpdateUser: false });
    await expect(first).resolves.toBe(false);
    expect(useAuthStore.getState().isValidating).toBe(true);

    resolveSecond?.({ user, isValid: true, shouldUpdateUser: false });
    await expect(second).resolves.toBe(true);
    expect(useAuthStore.getState().isValidating).toBe(false);
  });
});
