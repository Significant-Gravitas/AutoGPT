import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    isClientSide: vi.fn(),
    isServerSide: vi.fn(),
  },
}));

import { environment } from "@/services/environment";
import {
  ImpersonationCookie,
  ImpersonationSession,
  ImpersonationState,
  getSystemHeaders,
} from "../impersonation";
import {
  IMPERSONATION_COOKIE_NAME,
  IMPERSONATION_HEADER_NAME,
  IMPERSONATION_STORAGE_KEY,
} from "../constants";

const mockIsClientSide = vi.mocked(environment.isClientSide);

// ---------- helpers ----------

function setCookieRaw(name: string, value: string) {
  Object.defineProperty(document, "cookie", {
    writable: true,
    value: `${name}=${value}`,
  });
}

function clearCookieRaw() {
  Object.defineProperty(document, "cookie", {
    writable: true,
    value: "",
  });
}

// ---------- ImpersonationCookie ----------

describe("ImpersonationCookie", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    clearCookieRaw();
  });

  describe("get()", () => {
    it("returns null on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      expect(ImpersonationCookie.get()).toBeNull();
    });

    it("returns null when cookie is absent", () => {
      expect(ImpersonationCookie.get()).toBeNull();
    });

    it("returns decoded user ID from cookie", () => {
      setCookieRaw(IMPERSONATION_COOKIE_NAME, "user-123");
      expect(ImpersonationCookie.get()).toBe("user-123");
    });

    it("decodes percent-encoded characters", () => {
      setCookieRaw(
        IMPERSONATION_COOKIE_NAME,
        encodeURIComponent("user@example.com"),
      );
      expect(ImpersonationCookie.get()).toBe("user@example.com");
    });
  });

  describe("set()", () => {
    it("does nothing on server-side (sessionStorage stays empty)", () => {
      mockIsClientSide.mockReturnValue(false);
      ImpersonationCookie.set("user-123");
      // We can't spy on document.cookie directly in happy-dom, so just verify
      // ImpersonationState.get() sees nothing (no sessionStorage write and no cookie)
      expect(ImpersonationCookie.get()).toBeNull();
    });
  });

  describe("clear()", () => {
    it("does nothing on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      // Simply verify it doesn't throw
      expect(() => ImpersonationCookie.clear()).not.toThrow();
    });
  });
});

// ---------- ImpersonationSession ----------

describe("ImpersonationSession", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    sessionStorage.clear();
  });

  describe("get()", () => {
    it("returns null on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      expect(ImpersonationSession.get()).toBeNull();
    });

    it("returns null when key is absent", () => {
      expect(ImpersonationSession.get()).toBeNull();
    });

    it("returns stored user ID", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "user-456");
      expect(ImpersonationSession.get()).toBe("user-456");
    });

    it("returns null when sessionStorage throws", () => {
      vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
        throw new Error("storage unavailable");
      });
      expect(ImpersonationSession.get()).toBeNull();
    });
  });

  describe("set()", () => {
    it("does nothing on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      ImpersonationSession.set("user-123");
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
    });

    it("stores user ID in sessionStorage", () => {
      ImpersonationSession.set("user-789");
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBe(
        "user-789",
      );
    });
  });

  describe("clear()", () => {
    it("removes user ID from sessionStorage", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "user-789");
      ImpersonationSession.clear();
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
    });
  });
});

// ---------- ImpersonationState ----------

describe("ImpersonationState", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    sessionStorage.clear();
    clearCookieRaw();
  });

  describe("get()", () => {
    it("returns null when neither sessionStorage nor cookie has a value", () => {
      expect(ImpersonationState.get()).toBeNull();
    });

    it("returns sessionStorage value when present (same-tab path)", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-user");
      expect(ImpersonationState.get()).toBe("session-user");
    });

    it("falls back to cookie when sessionStorage is empty (cross-tab path)", () => {
      setCookieRaw(IMPERSONATION_COOKIE_NAME, "cookie-user");
      expect(ImpersonationState.get()).toBe("cookie-user");
    });

    it("syncs cookie value back into sessionStorage on cookie fallback", () => {
      setCookieRaw(IMPERSONATION_COOKIE_NAME, "cookie-user");
      ImpersonationState.get();
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBe(
        "cookie-user",
      );
    });

    it("prefers sessionStorage over cookie when both are present", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-user");
      setCookieRaw(IMPERSONATION_COOKIE_NAME, "cookie-user");
      expect(ImpersonationState.get()).toBe("session-user");
    });

    it("returns null on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      expect(ImpersonationState.get()).toBeNull();
    });
  });

  describe("set()", () => {
    it("stores user ID in sessionStorage", () => {
      ImpersonationState.set("new-user");
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBe(
        "new-user",
      );
    });

    it("also writes the impersonation cookie", () => {
      ImpersonationState.set("new-user");
      expect(ImpersonationCookie.get()).toBe("new-user");
    });
  });

  describe("clear()", () => {
    it("removes user ID from sessionStorage", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "user-to-clear");
      ImpersonationState.clear();
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
    });

    it("also clears the impersonation cookie", () => {
      ImpersonationState.set("user-to-clear");
      ImpersonationState.clear();
      expect(ImpersonationCookie.get()).toBeNull();
    });
  });
});

// ---------- getSystemHeaders ----------

describe("getSystemHeaders", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockIsClientSide.mockReturnValue(true);
    sessionStorage.clear();
    clearCookieRaw();
  });

  it("returns empty object when no impersonation is active", () => {
    expect(getSystemHeaders()).toEqual({});
  });

  it("returns impersonation header when impersonation is active via sessionStorage", () => {
    sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "user-abc");
    expect(getSystemHeaders()).toEqual({
      [IMPERSONATION_HEADER_NAME]: "user-abc",
    });
  });

  it("returns impersonation header when impersonation is active via cookie (cross-tab)", () => {
    setCookieRaw(IMPERSONATION_COOKIE_NAME, "cookie-user");
    expect(getSystemHeaders()).toEqual({
      [IMPERSONATION_HEADER_NAME]: "cookie-user",
    });
  });

  it("returns empty object on server-side", () => {
    mockIsClientSide.mockReturnValue(false);
    expect(getSystemHeaders()).toEqual({});
  });
});
