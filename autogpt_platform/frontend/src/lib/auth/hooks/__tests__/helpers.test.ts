import type BackendAPI from "@/lib/autogpt-server-api/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { User } from "../../types";

const getCurrentUserMock = vi.fn();
const validateSessionActionMock = vi.fn();
const refreshSessionActionMock = vi.fn();
const clearWebSocketDisconnectIntentMock = vi.fn();
const setWebSocketDisconnectIntentMock = vi.fn();
const getRedirectPathMock = vi.fn();
const isLogoutEventMock = vi.fn();

vi.mock("../../actions", () => ({
  getCurrentUser: () => getCurrentUserMock(),
  validateSession: (...args: unknown[]) => validateSessionActionMock(...args),
  refreshSession: () => refreshSessionActionMock(),
}));

vi.mock("../../helpers", () => ({
  clearWebSocketDisconnectIntent: () => clearWebSocketDisconnectIntentMock(),
  setWebSocketDisconnectIntent: () => setWebSocketDisconnectIntentMock(),
  getRedirectPath: (...args: unknown[]) => getRedirectPathMock(...args),
  isLogoutEvent: (...args: unknown[]) => isLogoutEventMock(...args),
}));

import {
  fetchUser,
  handleStorageEvent,
  refreshSession,
  validateSession,
} from "../helpers";

function makeUser(id: string): User {
  return {
    id,
    email: `${id}@example.com`,
    role: "authenticated",
    user_metadata: {},
  };
}

beforeEach(() => {
  getCurrentUserMock.mockReset();
  validateSessionActionMock.mockReset();
  refreshSessionActionMock.mockReset();
  clearWebSocketDisconnectIntentMock.mockReset();
  setWebSocketDisconnectIntentMock.mockReset();
  getRedirectPathMock.mockReset();
  isLogoutEventMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("fetchUser", () => {
  it("returns the user and clears the websocket disconnect intent", async () => {
    const user = makeUser("user-1");
    getCurrentUserMock.mockResolvedValue({ user });

    const result = await fetchUser();

    expect(result).toEqual({
      user,
      hasLoadedUser: true,
      isUserLoading: false,
    });
    expect(clearWebSocketDisconnectIntentMock).toHaveBeenCalled();
  });

  it("keeps hasLoadedUser false when there is no user and no error", async () => {
    getCurrentUserMock.mockResolvedValue({ user: null });

    const result = await fetchUser();

    expect(result).toEqual({
      user: null,
      hasLoadedUser: false,
      isUserLoading: false,
    });
    expect(clearWebSocketDisconnectIntentMock).not.toHaveBeenCalled();
  });

  it("marks the user as loaded when the action reports an error", async () => {
    getCurrentUserMock.mockResolvedValue({ user: null, error: "boom" });

    const result = await fetchUser();

    expect(result).toEqual({
      user: null,
      hasLoadedUser: true,
      isUserLoading: false,
    });
  });

  it("marks the user as loaded when the action throws", async () => {
    getCurrentUserMock.mockRejectedValue(new Error("network down"));

    const result = await fetchUser();

    expect(result).toEqual({
      user: null,
      hasLoadedUser: true,
      isUserLoading: false,
    });
  });
});

describe("validateSession", () => {
  it("returns the redirect path and requests a user update for invalid sessions", async () => {
    validateSessionActionMock.mockResolvedValue({
      user: null,
      isValid: false,
      redirectPath: "/login?next=%2Fcopilot",
    });

    const result = await validateSession({
      path: "/copilot",
      currentUser: makeUser("user-1"),
    });

    expect(result).toEqual({
      isValid: false,
      redirectPath: "/login?next=%2Fcopilot",
      shouldUpdateUser: true,
    });
  });

  it("requests a user update when the session user changed", async () => {
    const newUser = makeUser("user-2");
    validateSessionActionMock.mockResolvedValue({
      user: newUser,
      isValid: true,
    });

    const result = await validateSession({
      path: "/copilot",
      currentUser: makeUser("user-1"),
    });

    expect(result).toEqual({
      isValid: true,
      user: newUser,
      shouldUpdateUser: true,
    });
    expect(clearWebSocketDisconnectIntentMock).toHaveBeenCalled();
  });

  it("skips the user update when the session user is unchanged", async () => {
    const user = makeUser("user-1");
    validateSessionActionMock.mockResolvedValue({ user, isValid: true });

    const result = await validateSession({
      path: "/copilot",
      currentUser: user,
    });

    expect(result.shouldUpdateUser).toBe(false);
    expect(result.isValid).toBe(true);
  });

  it("stays valid without updating when the action returns no user", async () => {
    validateSessionActionMock.mockResolvedValue({ user: null, isValid: true });

    const result = await validateSession({
      path: "/copilot",
      currentUser: null,
    });

    expect(result).toEqual({ isValid: true, shouldUpdateUser: false });
  });

  it("falls back to the helper redirect path when the action throws", async () => {
    validateSessionActionMock.mockRejectedValue(new Error("server gone"));
    getRedirectPathMock.mockReturnValue("/login?next=%2Fcopilot");

    const result = await validateSession({
      path: "/copilot",
      currentUser: makeUser("user-1"),
    });

    expect(getRedirectPathMock).toHaveBeenCalledWith("/copilot");
    expect(result).toEqual({
      isValid: false,
      redirectPath: "/login?next=%2Fcopilot",
      shouldUpdateUser: true,
    });
  });
});

describe("refreshSession", () => {
  it("clears the websocket disconnect intent when a user is returned", async () => {
    const user = makeUser("user-1");
    refreshSessionActionMock.mockResolvedValue({ user });

    const result = await refreshSession();

    expect(result).toEqual({ user });
    expect(clearWebSocketDisconnectIntentMock).toHaveBeenCalled();
  });

  it("leaves the websocket disconnect intent alone when no user is returned", async () => {
    refreshSessionActionMock.mockResolvedValue({
      user: null,
      error: "No active session",
    });

    const result = await refreshSession();

    expect(result).toEqual({ user: null, error: "No active session" });
    expect(clearWebSocketDisconnectIntentMock).not.toHaveBeenCalled();
  });
});

describe("handleStorageEvent", () => {
  it("ignores storage events that are not logout broadcasts", () => {
    isLogoutEventMock.mockReturnValue(false);
    const disconnectWebSocket = vi.fn();
    const api = { disconnectWebSocket } as unknown as BackendAPI;

    const result = handleStorageEvent({
      event: new StorageEvent("storage", { key: "theme" }),
      api,
      router: null,
      path: "/copilot",
    });

    expect(result).toEqual({ shouldLogout: false });
    expect(setWebSocketDisconnectIntentMock).not.toHaveBeenCalled();
    expect(disconnectWebSocket).not.toHaveBeenCalled();
  });

  it("disconnects the websocket and returns the redirect path on logout", () => {
    isLogoutEventMock.mockReturnValue(true);
    getRedirectPathMock.mockReturnValue("/login?next=%2Fcopilot");
    const disconnectWebSocket = vi.fn();
    const api = { disconnectWebSocket } as unknown as BackendAPI;

    const result = handleStorageEvent({
      event: new StorageEvent("storage", { key: "supabase-logout" }),
      api,
      router: null,
      path: "/copilot",
    });

    expect(setWebSocketDisconnectIntentMock).toHaveBeenCalled();
    expect(disconnectWebSocket).toHaveBeenCalled();
    expect(result).toEqual({
      shouldLogout: true,
      redirectPath: "/login?next=%2Fcopilot",
    });
  });

  it("handles logout without an api client", () => {
    isLogoutEventMock.mockReturnValue(true);
    getRedirectPathMock.mockReturnValue(null);

    const result = handleStorageEvent({
      event: new StorageEvent("storage", { key: "supabase-logout" }),
      api: null,
      router: null,
      path: "/marketplace",
    });

    expect(result).toEqual({ shouldLogout: true, redirectPath: null });
  });
});
