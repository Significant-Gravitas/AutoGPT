import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/services/environment", () => ({
  environment: {
    isClientSide: vi.fn(),
    isServerSide: vi.fn(),
  },
}));

import { environment } from "@/services/environment";
import {
  IMPERSONATION_COOKIE_NAME,
  IMPERSONATION_HEADER_NAME,
  IMPERSONATION_STORAGE_KEY,
} from "../constants";
import {
  decodeImpersonation,
  encodeImpersonation,
  getSystemHeaders,
  ImpersonationCookie,
  ImpersonationSession,
  ImpersonationState,
  isImpersonationUsableBy,
  setImpersonationViewer,
} from "../impersonation";

const mockIsClientSide = vi.mocked(environment.isClientSide);

const ADMIN = { id: "admin-1", role: "admin" };
const OTHER_ADMIN = { id: "admin-2", role: "admin" };
const record = (target: string, owner = ADMIN.id) => ({ target, owner });

function setCookiesRaw(cookies: Record<string, string>) {
  Object.defineProperty(document, "cookie", {
    configurable: true,
    writable: true,
    value: Object.entries(cookies)
      .map(([name, value]) => `${name}=${value}`)
      .join("; "),
  });
}

function cookieHolding(target: string) {
  return {
    [IMPERSONATION_COOKIE_NAME]: encodeURIComponent(
      encodeImpersonation(record(target)),
    ),
  };
}

function useRealCookieJar() {
  Reflect.deleteProperty(document, "cookie");
  ImpersonationCookie.clear();
}

beforeEach(() => {
  vi.clearAllMocks();
  mockIsClientSide.mockReturnValue(true);
  sessionStorage.clear();
  setCookiesRaw({});
  setImpersonationViewer(null);
});

describe("impersonation records", () => {
  it("round-trips a target together with its owner", () => {
    expect(decodeImpersonation(encodeImpersonation(record("t")))).toEqual(
      record("t"),
    );
  });

  it("rejects anything that isn't a whole record", () => {
    expect(decodeImpersonation("bare-user-id")).toBeNull();
    expect(decodeImpersonation(JSON.stringify({ target: "t" }))).toBeNull();
    expect(decodeImpersonation(JSON.stringify({ owner: "o" }))).toBeNull();
    expect(decodeImpersonation(null)).toBeNull();
  });
});

describe("ImpersonationState.read()", () => {
  it("follows the shared cookie when another tab chose a different target", () => {
    ImpersonationSession.set(record("old-target"));
    setCookiesRaw(cookieHolding("new-target"));
    // The server checks requests against the cookie, so this tab must too.
    expect(ImpersonationState.read()).toEqual(record("new-target"));
    expect(ImpersonationSession.get()).toEqual(record("new-target"));
  });

  it("keeps this tab's copy when the cookie holds the same record", () => {
    ImpersonationSession.set(record("same-target"));
    setCookiesRaw(cookieHolding("same-target"));
    expect(ImpersonationState.read()).toEqual(record("same-target"));
  });

  it("falls back to the cookie and keeps a copy (cross-tab path)", () => {
    setCookiesRaw(cookieHolding("cookie-target"));
    expect(ImpersonationState.read()).toEqual(record("cookie-target"));
    expect(ImpersonationSession.get()).toEqual(record("cookie-target"));
  });

  it("drops a stored value that isn't a whole record (legacy bare id)", () => {
    sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "orphan");
    expect(ImpersonationState.read()).toBeNull();
    expect(sessionStorage.getItem(IMPERSONATION_STORAGE_KEY)).toBeNull();
  });

  it("never pairs this tab's target with another source's owner", () => {
    sessionStorage.setItem(IMPERSONATION_STORAGE_KEY, "session-target");
    setCookiesRaw(cookieHolding("cookie-target"));
    // The cookie's record, whole; never session-target with the cookie's owner.
    expect(ImpersonationState.read()).toEqual(record("cookie-target"));
  });

  it("drops a tab copy that no cookie backs (cleared, or from an old session)", () => {
    ImpersonationSession.set(record("restored-target"));
    expect(ImpersonationState.read()).toBeNull();
    expect(ImpersonationSession.getRaw()).toBeNull();
    expect(isImpersonationUsableBy(ADMIN)).toBe(false);
  });

  it("returns null on server-side", () => {
    mockIsClientSide.mockReturnValue(false);
    expect(ImpersonationState.read()).toBeNull();
  });
});

describe("getSystemHeaders", () => {
  it("sends the target only for the signed-in admin who chose it", () => {
    setCookiesRaw(cookieHolding("user-abc"));
    setImpersonationViewer(ADMIN);
    expect(getSystemHeaders()).toEqual({
      [IMPERSONATION_HEADER_NAME]: "user-abc",
    });
  });

  it("sends nothing for a different admin, however the record got there", () => {
    setCookiesRaw(cookieHolding("user-abc"));
    setImpersonationViewer(OTHER_ADMIN);
    expect(getSystemHeaders()).toEqual({});
  });

  it("sends nothing for the owner once they're no longer an admin", () => {
    setCookiesRaw(cookieHolding("user-abc"));
    setImpersonationViewer({ id: ADMIN.id, role: "authenticated" });
    expect(getSystemHeaders()).toEqual({});
  });

  it("sends nothing while nobody is signed in", () => {
    setCookiesRaw(cookieHolding("user-abc"));
    expect(getSystemHeaders()).toEqual({});
  });

  it("sends nothing when no impersonation is active", () => {
    setImpersonationViewer(ADMIN);
    expect(getSystemHeaders()).toEqual({});
  });
});

describe("writing and clearing", () => {
  beforeEach(useRealCookieJar);

  it("stores the target and owner as one value in sessionStorage and the cookie", () => {
    ImpersonationState.set("new-user", ADMIN.id);
    expect(ImpersonationSession.get()).toEqual(record("new-user"));
    expect(ImpersonationCookie.get()).toEqual(record("new-user"));
  });

  it("clears both copies", () => {
    ImpersonationState.set("user-to-clear", ADMIN.id);
    ImpersonationState.clear();
    expect(ImpersonationSession.getRaw()).toBeNull();
    expect(ImpersonationCookie.getRaw()).toBeNull();
  });

  it("does nothing on server-side", () => {
    mockIsClientSide.mockReturnValue(false);
    ImpersonationState.set("user-123", ADMIN.id);
    mockIsClientSide.mockReturnValue(true);
    expect(ImpersonationState.read()).toBeNull();
  });
});

describe("server-side reads", () => {
  const cookieWith = (value: string | null) => async (name: string) =>
    name === IMPERSONATION_COOKIE_NAME ? value : null;

  beforeEach(() => {
    mockIsClientSide.mockReturnValue(false);
  });

  it("returns the target when the session is the admin who chose it", async () => {
    expect(
      await ImpersonationState.getServerSide(
        cookieWith(encodeImpersonation(record("user-a"))),
        async () => ADMIN,
      ),
    ).toBe("user-a");
  });

  it("refuses another admin's target", async () => {
    expect(
      await ImpersonationState.getServerSide(
        cookieWith(encodeImpersonation(record("user-a"))),
        async () => OTHER_ADMIN,
      ),
    ).toBeNull();
  });

  it("refuses a non-admin session and a missing session", async () => {
    const cookie = cookieWith(encodeImpersonation(record("user-a")));
    expect(
      await ImpersonationState.getServerSide(cookie, async () => ({
        id: ADMIN.id,
        role: "authenticated",
      })),
    ).toBeNull();
    expect(
      await ImpersonationState.getServerSide(cookie, async () => null),
    ).toBeNull();
  });

  it("refuses a cookie that isn't a whole record", async () => {
    expect(
      await ImpersonationState.getServerSide(
        cookieWith("user-a"),
        async () => ADMIN,
      ),
    ).toBeNull();
  });
});
