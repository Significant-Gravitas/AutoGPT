import type BackendAPI from "@/lib/autogpt-server-api/client";
import { getSystemHeaders, ImpersonationState } from "@/lib/impersonation";
import { SessionKey } from "@/services/storage/session-storage";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "../../types";

// Only the server actions are faked; the store, auth helpers, impersonation
// state and header builder run as shipped.
const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("../../actions", () => actions);
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear: vi.fn() }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));
const toast = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", () => ({ toast }));

import { useAuthStore } from "../useAuthStore";

const ACT_AS = "X-Act-As-User-Id";
const admin = {
  id: "admin-1",
  email: "admin@example.com",
  role: "admin",
} as User;

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

function validate(router = makeRouter()) {
  return useAuthStore
    .getState()
    .validateSession({ router, path: "/library", force: true });
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  ImpersonationState.clear();
  ImpersonationState.set("user-a", admin.id);
});

afterEach(() => {
  useAuthStore.getState().cleanup();
  useAuthStore.setState({
    user: null,
    hasLoadedUser: false,
    isUserLoading: true,
    isValidating: false,
    lastValidation: 0,
    routerRef: null,
    apiRef: null,
    currentPath: "",
  });
  ImpersonationState.clear();
  vi.clearAllMocks();
});

describe("a session check that can't get an answer changes nothing", () => {
  it("keeps the admin and their target when the check never reaches the server", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
    actions.validateSession.mockRejectedValue(new TypeError("Load failed"));
    const router = makeRouter();

    await validate(router);

    expect(useAuthStore.getState().user).toBe(admin);
    expect(router.push).not.toHaveBeenCalled();
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("keeps them when the auth backend reports itself unavailable", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      isUnavailable: true,
    });

    await validate();

    expect(useAuthStore.getState().user).toBe(admin);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("keeps the target when a tab starts up while the backend is unavailable", async () => {
    actions.getCurrentUser.mockResolvedValue({ user: null });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      isUnavailable: true,
    });

    await useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router: makeRouter(),
      path: "/library",
    });

    // Kept for when the admin is known again; nothing is sent meanwhile,
    // because nobody is known to be signed in.
    expect(ImpersonationState.read()).toEqual({
      target: "user-a",
      owner: admin.id,
    });
    expect(getSystemHeaders()).toEqual({});
  });

  it("keeps the admin signed in when a refresh can't reach the backend", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    actions.refreshSession.mockResolvedValue({
      user: null,
      error: "connection reset",
      isUnavailable: true,
    });

    await useAuthStore.getState().refreshSession();

    expect(useAuthStore.getState().user).toBe(admin);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });
});

describe("a deploy skew the reload didn't fix", () => {
  const reload = vi.fn();
  const nativeReload = Object.getOwnPropertyDescriptor(
    window.location,
    "reload",
  );

  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(window.location, "reload", {
      configurable: true,
      value: reload,
    });
  });

  afterEach(() => {
    if (nativeReload) {
      Object.defineProperty(window.location, "reload", nativeReload);
    }
    vi.useRealTimers();
  });

  it("asks for a refresh rather than reloading again when the retries outlast the cooldown", async () => {
    // This page is the one the skew reload loaded a moment ago.
    sessionStorage.setItem(
      SessionKey.DEPLOYMENT_RELOAD_AT,
      String(Date.now() - 5_000),
    );
    // Retries run against the page the store last saw, as in the app.
    useAuthStore.setState({
      user: admin,
      hasLoadedUser: true,
      routerRef: makeRouter(),
      currentPath: "/library",
    });
    actions.validateSession.mockRejectedValue(
      Object.assign(new Error("Server Action was not found on the server."), {
        name: "UnrecognizedActionError",
      }),
    );

    await validate();
    // Every retry of the backoff (1 + 2 + 4 + 8 + 16 + 30 s) meets the skew.
    await vi.advanceTimersByTimeAsync(120_000);

    expect(actions.validateSession).toHaveBeenCalledTimes(7);
    expect(reload).not.toHaveBeenCalled();
    expect(toast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "A new version of AutoGPT is available",
      }),
    );
    expect(useAuthStore.getState().user).toBe(admin);
  });
});
