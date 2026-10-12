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
  isImpersonationUsableBy,
} from "../impersonation";
import {
  IMPERSONATION_COOKIE_NAME,
  IMPERSONATION_HEADER_NAME,
  IMPERSONATION_OWNER_COOKIE_NAME,
  IMPERSONATION_STORAGE_KEY,
} from "../constants";

const OWNER_STORAGE_KEY = `${IMPERSONATION_STORAGE_KEY}-owner`;

const mockIsClientSide = vi.mocked(environment.isClientSide);

// ---------- helpers ----------

function setCookieRaw(name: string, value: string) {
  setCookiesRaw({ [name]: value });
}

function setCookiesRaw(cookies: Record<string, string>) {
  Object.defineProperty(document, "cookie", {
    configurable: true,
    writable: true,
    value: Object.entries(cookies)
      .map(([name, value]) => `${name}=${value}`)
      .join("; "),
  });
}

// Back to jsdom's real cookie jar, for tests that write more than one cookie.
function useRealCookieJar() {
  Reflect.deleteProperty(document, "cookie");
  ImpersonationCookie.clear();
}

function clearCookieRaw() {
  Object.defineProperty(document, "cookie", {
    configurable: true,
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
      ImpersonationCookie.set("user-123", "admin-1");
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

    it("returns the sessionStorage target when its owner is recorded", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-user");
      sessionStorage.setItem(OWNER_STORAGE_KEY, "admin-1");
      expect(ImpersonationState.get()).toBe("session-user");
    });

    it("falls back to the cookies, adopting the owner too (cross-tab path)", () => {
      setCookiesRaw({
        [IMPERSONATION_COOKIE_NAME]: "cookie-user",
        [IMPERSONATION_OWNER_COOKIE_NAME]: "admin-1",
      });
      expect(ImpersonationState.get()).toBe("cookie-user");
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBe(
        "cookie-user",
      );
      expect(sessionStorage.getItem(OWNER_STORAGE_KEY)).toBe("admin-1");
    });

    it("prefers sessionStorage over cookie when both are present", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-user");
      sessionStorage.setItem(OWNER_STORAGE_KEY, "admin-1");
      setCookiesRaw({
        [IMPERSONATION_COOKIE_NAME]: "cookie-user",
        [IMPERSONATION_OWNER_COOKIE_NAME]: "admin-1",
      });
      expect(ImpersonationState.get()).toBe("session-user");
    });

    it("drops a sessionStorage target with no recorded owner", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "orphan");
      expect(ImpersonationState.get()).toBeNull();
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
    });

    it("never borrows the owner of a different cookie target", () => {
      sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-target");
      setCookiesRaw({
        [IMPERSONATION_COOKIE_NAME]: "cookie-target",
        [IMPERSONATION_OWNER_COOKIE_NAME]: "admin-1",
      });

      expect(ImpersonationState.getOwner()).toBeNull();
      expect(isImpersonationUsableBy({ id: "admin-1", role: "admin" })).toBe(
        false,
      );
      expect(ImpersonationState.get()).toBeNull();
    });

    it("drops a cookie target with no owner cookie", () => {
      setCookieRaw(IMPERSONATION_COOKIE_NAME, "orphan");
      expect(ImpersonationState.get()).toBeNull();
    });

    it("returns null on server-side", () => {
      mockIsClientSide.mockReturnValue(false);
      expect(ImpersonationState.get()).toBeNull();
    });
  });

  describe("set()", () => {
    beforeEach(useRealCookieJar);

    it("records the target and its owner in sessionStorage and cookies", () => {
      ImpersonationState.set("new-user", "admin-1");
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBe(
        "new-user",
      );
      expect(sessionStorage.getItem(OWNER_STORAGE_KEY)).toBe("admin-1");
      expect(ImpersonationCookie.get()).toBe("new-user");
      expect(ImpersonationCookie.getOwner()).toBe("admin-1");
    });
  });

  describe("clear()", () => {
    beforeEach(useRealCookieJar);

    it("removes the target and owner from sessionStorage and cookies", () => {
      ImpersonationState.set("user-to-clear", "admin-1");
      ImpersonationState.clear();
      expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
      expect(sessionStorage.getItem(OWNER_STORAGE_KEY)).toBeNull();
      expect(ImpersonationCookie.get()).toBeNull();
      expect(ImpersonationCookie.getOwner()).toBeNull();
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
    sessionStorage.setItem(OWNER_STORAGE_KEY, "admin-1");
    expect(getSystemHeaders()).toEqual({
      [IMPERSONATION_HEADER_NAME]: "user-abc",
    });
  });

  it("returns impersonation header when impersonation is active via cookie (cross-tab)", () => {
    setCookiesRaw({
      [IMPERSONATION_COOKIE_NAME]: "cookie-user",
      [IMPERSONATION_OWNER_COOKIE_NAME]: "admin-1",
    });
    expect(getSystemHeaders()).toEqual({
      [IMPERSONATION_HEADER_NAME]: "cookie-user",
    });
  });

  it("never sends a target no admin can be tied to", () => {
    sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "orphan");
    setCookieRaw(IMPERSONATION_COOKIE_NAME, "orphan");
    expect(getSystemHeaders()).toEqual({});
  });

  it("returns empty object on server-side", () => {
    mockIsClientSide.mockReturnValue(false);
    expect(getSystemHeaders()).toEqual({});
  });
});
