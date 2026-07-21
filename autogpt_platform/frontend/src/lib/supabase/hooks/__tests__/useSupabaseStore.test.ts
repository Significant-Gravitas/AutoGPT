import type BackendAPI from "@/lib/autogpt-server-api/client";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

// The store imports a chain of modules (server actions, supabase helpers,
// react-query). Stub them so we can exercise `setCurrentRequestContext` —
// the new behavior introduced by this PR — in isolation, without pulling
// in next/headers or a real Supabase client at test time.
vi.mock("../../actions", () => ({
  serverLogout: vi.fn(),
}));
vi.mock("../../helpers", () => ({
  broadcastLogout: vi.fn(),
  setWebSocketDisconnectIntent: vi.fn(),
  setupSessionEventListeners: vi.fn(() => ({ cleanup: vi.fn() })),
}));
vi.mock("../helpers", () => ({
  ensureSupabaseClient: vi.fn(() => null),
  fetchUser: vi.fn(),
  handleStorageEvent: vi.fn(),
  refreshSession: vi.fn(),
  validateSession: vi.fn(),
}));

import type { User } from "@supabase/supabase-js";
import { validateSession as validateSessionHelper } from "../helpers";
import { useSupabaseStore } from "../useSupabaseStore";

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
  const { routerRef, apiRef, currentPath } = useSupabaseStore.getState();
  return { routerRef, apiRef, currentPath };
}

describe("useSupabaseStore.setCurrentRequestContext", () => {
  afterEach(() => {
    useSupabaseStore.setState({
      routerRef: null,
      apiRef: null,
      currentPath: "",
    });
  });

  it("writes refs and path to the store on first call", () => {
    const router = makeRouter();
    const api = makeApi();

    useSupabaseStore
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
    const { setCurrentRequestContext } = useSupabaseStore.getState();

    setCurrentRequestContext({ router, api, path: "/library" });
    const setSpy = vi.spyOn(useSupabaseStore, "setState");

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

    const { setCurrentRequestContext } = useSupabaseStore.getState();
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

describe("useSupabaseStore.validateSession", () => {
  afterEach(() => {
    useSupabaseStore.setState({
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
    useSupabaseStore.setState({
      user: { id: "user-1" } as User,
      hasLoadedUser: true,
    });

    const stillValid = await useSupabaseStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(stillValid).toBe(false);
    expect(useSupabaseStore.getState().user).toBeNull();
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
    useSupabaseStore.setState({ user, hasLoadedUser: true });

    const stillValid = await useSupabaseStore.getState().validateSession({
      router,
      path: "/build",
      force: true,
    });

    expect(stillValid).toBe(true);
    expect(useSupabaseStore.getState().user).toBe(user);
    expect(router.push).not.toHaveBeenCalled();
  });
});
