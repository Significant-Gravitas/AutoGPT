import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const storageGet = vi.fn();
const storageSet = vi.fn();
const storageClean = vi.fn();
const isServerSideMock = vi.fn();

vi.mock("@/services/storage/local-storage", () => ({
  Key: {
    LOGOUT: "supabase-logout",
    WEBSOCKET_DISCONNECT_INTENT: "websocket-disconnect-intent",
  },
  storage: {
    get: (...args: unknown[]) => storageGet(...args),
    set: (...args: unknown[]) => storageSet(...args),
    clean: (...args: unknown[]) => storageClean(...args),
  },
}));

vi.mock("@/services/environment", () => ({
  environment: {
    isServerSide: () => isServerSideMock(),
  },
}));

import {
  broadcastLogout,
  clearWebSocketDisconnectIntent,
  getRedirectPath,
  hasWebSocketDisconnectIntent,
  isAdminPage,
  isLogoutEvent,
  isProtectedPage,
  setupSessionEventListeners,
  setWebSocketDisconnectIntent,
} from "../helpers";

beforeEach(() => {
  storageGet.mockReset();
  storageSet.mockReset();
  storageClean.mockReset();
  isServerSideMock.mockReset().mockReturnValue(false);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("isProtectedPage", () => {
  it("matches a protected page exactly", () => {
    expect(isProtectedPage("/copilot")).toBe(true);
  });

  it("matches nested paths under a protected prefix", () => {
    expect(isProtectedPage("/library/agents/123")).toBe(true);
  });

  it("rejects public pages", () => {
    expect(isProtectedPage("/marketplace")).toBe(false);
  });
});

describe("isAdminPage", () => {
  it("matches the admin section and its sub-pages", () => {
    expect(isAdminPage("/admin")).toBe(true);
    expect(isAdminPage("/admin/marketplace")).toBe(true);
  });

  it("rejects non-admin pages", () => {
    expect(isAdminPage("/profile")).toBe(false);
  });
});

describe("getRedirectPath", () => {
  it("sends protected paths to login preserving the original path", () => {
    expect(getRedirectPath("/copilot?tab=chat")).toBe(
      `/login?next=${encodeURIComponent("/copilot?tab=chat")}`,
    );
  });

  it("sends admin paths through the logout redirect even for non-admin roles", () => {
    // isAdminPage is part of shouldRedirectOnLogout, so admin paths hit the
    // login redirect before the role check is ever reached.
    expect(getRedirectPath("/admin", "user")).toBe(
      `/login?next=${encodeURIComponent("/admin")}`,
    );
  });

  it("returns null for public paths", () => {
    expect(getRedirectPath("/marketplace")).toBeNull();
  });
});

describe("broadcastLogout", () => {
  it("writes a logout timestamp to the logout storage key", () => {
    broadcastLogout();

    expect(storageSet).toHaveBeenCalledWith(
      "supabase-logout",
      expect.stringMatching(/^\d+$/),
    );
  });
});

describe("isLogoutEvent", () => {
  it("recognizes storage events for the logout key", () => {
    const event = new StorageEvent("storage", { key: "supabase-logout" });

    expect(isLogoutEvent(event)).toBe(true);
  });

  it("ignores storage events for other keys", () => {
    const event = new StorageEvent("storage", { key: "theme" });

    expect(isLogoutEvent(event)).toBe(false);
  });
});

describe("webSocket disconnect intent", () => {
  it("stores the disconnect intent flag", () => {
    setWebSocketDisconnectIntent();

    expect(storageSet).toHaveBeenCalledWith(
      "websocket-disconnect-intent",
      "true",
    );
  });

  it("clears the disconnect intent flag", () => {
    clearWebSocketDisconnectIntent();

    expect(storageClean).toHaveBeenCalledWith("websocket-disconnect-intent");
  });

  it("reports intent only when the stored flag is true", () => {
    storageGet.mockReturnValue("true");
    expect(hasWebSocketDisconnectIntent()).toBe(true);

    storageGet.mockReturnValue(undefined);
    expect(hasWebSocketDisconnectIntent()).toBe(false);
  });
});

describe("setupSessionEventListeners", () => {
  it("registers visibility and storage listeners and removes them on cleanup", () => {
    const documentAdd = vi.spyOn(document, "addEventListener");
    const documentRemove = vi.spyOn(document, "removeEventListener");
    const windowAdd = vi.spyOn(window, "addEventListener");
    const windowRemove = vi.spyOn(window, "removeEventListener");
    const onVisibilityChange = vi.fn();
    const onStorageChange = vi.fn();

    const listeners = setupSessionEventListeners(
      onVisibilityChange,
      onStorageChange,
    );

    expect(documentAdd).toHaveBeenCalledWith(
      "visibilitychange",
      onVisibilityChange,
    );
    expect(windowAdd).toHaveBeenCalledWith("storage", onStorageChange);

    listeners.cleanup();

    expect(documentRemove).toHaveBeenCalledWith(
      "visibilitychange",
      onVisibilityChange,
    );
    expect(windowRemove).toHaveBeenCalledWith("storage", onStorageChange);
  });

  it("returns a no-op cleanup without registering listeners on the server", () => {
    isServerSideMock.mockReturnValue(true);
    const documentAdd = vi.spyOn(document, "addEventListener");
    const windowAdd = vi.spyOn(window, "addEventListener");

    const listeners = setupSessionEventListeners(vi.fn(), vi.fn());

    expect(documentAdd).not.toHaveBeenCalled();
    expect(windowAdd).not.toHaveBeenCalled();
    expect(() => listeners.cleanup()).not.toThrow();
  });
});
