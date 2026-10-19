import type BackendAPI from "@/lib/autogpt-server-api/client";
import { IMPERSONATION_STORAGE_KEY } from "@/lib/constants";
import {
  getSystemHeaders,
  ImpersonationSession,
  ImpersonationState,
} from "@/lib/impersonation";
import type { AppRouterInstance } from "next/dist/shared/lib/app-router-context.shared-runtime";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "../../types";

// Only the server actions are faked. The store, both auth helper modules, the
// impersonation state and the request header builder all run for real, so
// these tests see exactly the header the next backend request would carry.
const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("../../actions", () => actions);
const reloadPage = vi.hoisted(() => vi.fn());
vi.mock("../../helpers", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../../helpers")>()),
  reloadPage,
}));
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear: vi.fn() }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));

import { useAuthStore } from "../useAuthStore";

const ACT_AS = "X-Act-As-User-Id";
const admin = {
  id: "admin-1",
  email: "admin@example.com",
  role: "admin",
} as User;
const otherAdmin = {
  id: "admin-2",
  email: "admin2@example.com",
  role: "admin",
} as User;
const otherUser = {
  id: "user-c",
  email: "c@example.com",
  role: "authenticated",
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

function validateAs(result: { user: User | null; isValid: boolean }) {
  actions.validateSession.mockResolvedValue({
    ...result,
    redirectPath: result.isValid ? undefined : "/login?next=%2Flibrary",
  });
  return useAuthStore
    .getState()
    .validateSession({ router: makeRouter(), path: "/library", force: true });
}

async function signInTab(user: User) {
  actions.getCurrentUser.mockResolvedValue({ user });
  await useAuthStore.getState().initialize({
    api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
    router: makeRouter(),
    path: "/library",
  });
}

function storageEvent(key: string) {
  window.dispatchEvent(
    new StorageEvent("storage", { key, newValue: Date.now().toString() }),
  );
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  ImpersonationState.clear();
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
    currentPath: "",
  });
  vi.clearAllMocks();
});

describe("admin impersonation follows the session that chose it", () => {
  it("is dropped, here and in other tabs, when the backend rejects the session", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a", admin.id);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });

    await validateAs({ user: null, isValid: false });

    expect(useAuthStore.getState().user).toBeNull();
    expect(getSystemHeaders()).toEqual({});
    expect(localStorage.getItem("impersonation-cleared")).not.toBeNull();
  });

  it("is dropped when a session refresh finds no session", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a", admin.id);
    actions.refreshSession.mockResolvedValue({
      user: null,
      error: "No active session",
    });

    await useAuthStore.getState().refreshSession();

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped when the tab finds itself signed in as someone else", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a", admin.id);

    await validateAs({ user: otherUser, isValid: true });

    expect(useAuthStore.getState().user).toBe(otherUser);
    expect(getSystemHeaders()).toEqual({});
  });

  it("stays while the same admin is still signed in", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a", admin.id);

    await validateAs({ user: { ...admin }, isValid: true });

    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("is dropped from this tab when another tab signs in as someone else", async () => {
    await signInTab(admin);
    // Acting as user-a until the other tab's sign-in clears it everywhere.
    ImpersonationState.set("user-a", admin.id);
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });

    storageEvent("impersonation-cleared");
    await validateAs({ user: otherUser, isValid: true });

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped from this tab when another tab signs out", async () => {
    await signInTab(admin);
    ImpersonationSession.set({ target: "user-a", owner: admin.id });

    storageEvent("supabase-logout");

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped everywhere on an explicit sign-out", async () => {
    await signInTab(admin);
    ImpersonationState.set("user-a", admin.id);
    actions.serverLogout.mockResolvedValue({ success: true });

    await useAuthStore.getState().logOut();

    expect(getSystemHeaders()).toEqual({});
    expect(localStorage.getItem("impersonation-cleared")).not.toBeNull();
  });

  describe("when a tab starts with no user, e.g. after a reload", () => {
    it("keeps the target for the admin who chose it", async () => {
      ImpersonationState.set("user-a", admin.id);

      await signInTab(admin);

      expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
    });

    it("drops it for an account that can't impersonate", async () => {
      ImpersonationSession.set({ target: "user-a", owner: admin.id });

      await signInTab(otherUser);

      expect(getSystemHeaders()).toEqual({});
    });

    it("drops it for a different admin", async () => {
      ImpersonationState.set("user-a", admin.id);

      await signInTab(otherAdmin);

      expect(getSystemHeaders()).toEqual({});
    });
  });

  it("is never sent from a restored tab that can't say whose it is", async () => {
    // A second tab copied the target from the cookie before owners were
    // recorded, missed the cleanup, and comes back signed in as another admin.
    sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "target-a");

    await signInTab(otherAdmin);

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped when the admin is demoted, even with the same account", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a", admin.id);

    await validateAs({
      user: { ...admin, role: "authenticated" },
      isValid: true,
    });

    expect(useAuthStore.getState().user?.role).toBe("authenticated");
    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped when a tab starts up and finds no session", async () => {
    ImpersonationState.set("user-a", admin.id);
    actions.getCurrentUser.mockResolvedValue({ user: null });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Flibrary",
    });

    await useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router: makeRouter(),
      path: "/library",
    });

    expect(getSystemHeaders()).toEqual({});
  });
});

describe("starting up with a session the backend no longer accepts", () => {
  async function startUpOn(path: string, redirectPath?: string) {
    const router = makeRouter();
    actions.getCurrentUser.mockResolvedValue({ user: null });
    actions.validateSession.mockResolvedValue({
      user: null,
      isValid: false,
      redirectPath,
    });
    await useAuthStore.getState().initialize({
      api: { disconnectWebSocket: vi.fn() } as unknown as BackendAPI,
      router,
      path,
    });
    return router;
  }

  it("sends a protected page to sign in", async () => {
    const router = await startUpOn("/library", "/login?next=%2Flibrary");

    expect(router.replace).toHaveBeenCalledWith("/login?next=%2Flibrary");
    expect(useAuthStore.getState()).toMatchObject({
      user: null,
      isUserLoading: false,
    });
  });

  it("sends it to sign in and drops a left-over target", async () => {
    ImpersonationState.set("user-a", admin.id);

    const router = await startUpOn("/library", "/login?next=%2Flibrary");

    expect(router.replace).toHaveBeenCalledWith("/login?next=%2Flibrary");
    expect(ImpersonationState.read()).toBeNull();
  });

  it("leaves a public page where it is", async () => {
    const router = await startUpOn("/marketplace");

    expect(router.replace).not.toHaveBeenCalled();
    expect(useAuthStore.getState().isUserLoading).toBe(false);
  });
});

describe("answers that arrive after a sign-out", () => {
  it("a refresh started before the sign-out can't sign the tab back in", async () => {
    await signInTab(admin);
    let answer!: (value: unknown) => void;
    actions.refreshSession.mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    actions.serverLogout.mockResolvedValue({ success: true });

    const refreshing = useAuthStore.getState().refreshSession();
    await useAuthStore.getState().logOut();
    answer({ user: admin });
    await refreshing;

    expect(useAuthStore.getState().user).toBeNull();
  });

  it("a validation started before the sign-out can't sign the tab back in", async () => {
    await signInTab(admin);
    let answer!: (value: unknown) => void;
    actions.validateSession.mockReturnValue(
      new Promise((resolve) => {
        answer = resolve;
      }),
    );
    actions.serverLogout.mockResolvedValue({ success: true });

    const validating = useAuthStore
      .getState()
      .validateSession({ router: makeRouter(), path: "/library", force: true });
    await useAuthStore.getState().logOut();
    answer({ user: { ...admin, role: "authenticated" }, isValid: true });
    await validating;

    expect(useAuthStore.getState().user).toBeNull();
  });

  it("no target is sent for another admin, however it was stored", async () => {
    await signInTab(otherAdmin);
    ImpersonationState.set("target-a", admin.id);

    expect(getSystemHeaders()).toEqual({});
  });
});

describe("signing out", () => {
  const LOGOUT_SIGNAL = "supabase-logout";

  afterEach(() => {
    vi.useRealTimers();
  });

  it("tells other tabs only once the server has ended the session", async () => {
    await signInTab(admin);
    let finishServerLogout!: () => void;
    actions.serverLogout.mockReturnValue(
      new Promise<{ success: boolean }>((resolve) => {
        finishServerLogout = () => resolve({ success: true });
      }),
    );

    const loggingOut = useAuthStore.getState().logOut();
    await vi.waitFor(() => expect(actions.serverLogout).toHaveBeenCalled());
    // A tab told now would reload into the still-live session.
    expect(localStorage.getItem(LOGOUT_SIGNAL)).toBeNull();

    finishServerLogout();
    await loggingOut;

    expect(localStorage.getItem(LOGOUT_SIGNAL)).not.toBeNull();
    expect(useAuthStore.getState().user).toBeNull();
  });

  it("keeps the user signed in, and tells no other tab, when the sign-out never reaches the server", async () => {
    await signInTab(admin);
    vi.useFakeTimers();
    actions.serverLogout.mockRejectedValue(new TypeError("Failed to fetch"));

    const loggingOut = useAuthStore.getState().logOut();
    const outcome = loggingOut.catch((error: unknown) => error);
    await vi.runAllTimersAsync();

    expect(await outcome).toBeInstanceOf(TypeError);
    expect(actions.serverLogout).toHaveBeenCalledTimes(3);
    expect(localStorage.getItem(LOGOUT_SIGNAL)).toBeNull();
    expect(useAuthStore.getState().user).toBe(admin);
  });

  it("retries a sign-out that didn't get through, then completes", async () => {
    await signInTab(admin);
    vi.useFakeTimers();
    actions.serverLogout
      .mockRejectedValueOnce(new TypeError("Failed to fetch"))
      .mockResolvedValueOnce({ success: true });

    const loggingOut = useAuthStore.getState().logOut();
    await vi.runAllTimersAsync();
    await loggingOut;

    expect(actions.serverLogout).toHaveBeenCalledTimes(2);
    expect(localStorage.getItem(LOGOUT_SIGNAL)).not.toBeNull();
    expect(useAuthStore.getState().user).toBeNull();
  });

  it("is signed out even when local cleanup fails after the server confirmed", async () => {
    await signInTab(admin);
    useAuthStore.setState({
      apiRef: {
        disconnectWebSocket: () => {
          throw new Error("socket already gone");
        },
      } as unknown as BackendAPI,
    });
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => undefined);
    actions.serverLogout.mockResolvedValue({ success: true });

    await useAuthStore.getState().logOut();

    expect(useAuthStore.getState().user).toBeNull();
    expect(localStorage.getItem(LOGOUT_SIGNAL)).not.toBeNull();
    expect(consoleError).toHaveBeenCalledWith(
      "Signed out, but couldn't disconnect the WebSocket:",
      expect.any(Error),
    );
  });
});

describe("another tab starts impersonating someone else", () => {
  function changedInAnotherTab() {
    window.dispatchEvent(
      new StorageEvent("storage", {
        key: "impersonation-changed",
        newValue: String(Math.random()),
      }),
    );
  }

  it("reloads the owning admin's tab so its page matches the new target", async () => {
    await signInTab(admin);
    ImpersonationState.set("new-target", admin.id);
    reloadPage.mockClear();

    changedInAnotherTab();

    expect(reloadPage).toHaveBeenCalledTimes(1);
  });

  it("leaves a tab signed in as someone else alone", async () => {
    await signInTab(otherUser);
    ImpersonationState.set("new-target", admin.id);
    reloadPage.mockClear();

    changedInAnotherTab();

    expect(reloadPage).not.toHaveBeenCalled();
  });
});
