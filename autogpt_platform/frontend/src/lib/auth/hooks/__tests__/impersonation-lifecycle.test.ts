import type BackendAPI from "@/lib/autogpt-server-api/client";
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
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear: vi.fn() }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity: vi.fn(),
}));

import { useAuthStore } from "../useAuthStore";

const ACT_AS = "X-Act-As-User-Id";
const admin = { id: "admin-1", email: "admin@example.com" } as User;
const otherUser = { id: "user-c", email: "c@example.com" } as User;

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
    ImpersonationState.set("user-a");
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });

    await validateAs({ user: null, isValid: false });

    expect(useAuthStore.getState().user).toBeNull();
    expect(getSystemHeaders()).toEqual({});
    expect(localStorage.getItem("impersonation-cleared")).not.toBeNull();
  });

  it("is dropped when a session refresh finds no session", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a");
    actions.refreshSession.mockResolvedValue({
      user: null,
      error: "No active session",
    });

    await useAuthStore.getState().refreshSession();

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped when the tab finds itself signed in as someone else", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a");

    await validateAs({ user: otherUser, isValid: true });

    expect(useAuthStore.getState().user).toBe(otherUser);
    expect(getSystemHeaders()).toEqual({});
  });

  it("stays while the same admin is still signed in", async () => {
    useAuthStore.setState({ user: admin, hasLoadedUser: true });
    ImpersonationState.set("user-a");

    await validateAs({ user: { ...admin }, isValid: true });

    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });
  });

  it("is dropped from this tab when another tab signs in as someone else", async () => {
    await signInTab(admin);
    // This tab's own copy; the other tab's sign-in already cleared the
    // shared cookie and announced it.
    ImpersonationSession.set("user-a");
    expect(getSystemHeaders()).toEqual({ [ACT_AS]: "user-a" });

    storageEvent("impersonation-cleared");
    await validateAs({ user: otherUser, isValid: true });

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped from this tab when another tab signs out", async () => {
    await signInTab(admin);
    ImpersonationSession.set("user-a");

    storageEvent("supabase-logout");

    expect(getSystemHeaders()).toEqual({});
  });

  it("is dropped everywhere on an explicit sign-out", async () => {
    await signInTab(admin);
    ImpersonationState.set("user-a");
    actions.serverLogout.mockResolvedValue({ success: true });

    await useAuthStore.getState().logOut();

    expect(getSystemHeaders()).toEqual({});
    expect(localStorage.getItem("impersonation-cleared")).not.toBeNull();
  });
});
