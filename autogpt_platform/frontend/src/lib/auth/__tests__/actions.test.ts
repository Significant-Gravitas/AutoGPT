import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();
const getServerAuthTokenMock = vi.fn();
const signOutMock = vi.fn();
const { cookieJar } = vi.hoisted(() => ({
  cookieJar: new Map<string, string>(),
}));

vi.mock("../auth", () => ({
  auth: {
    api: {
      signOut: (...args: unknown[]) => signOutMock(...args),
    },
  },
}));

vi.mock("../server/getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

vi.mock("@/lib/autogpt-server-api/helpers", () => ({
  getServerAuthToken: () => getServerAuthTokenMock(),
}));

vi.mock("next/headers", () => ({
  headers: vi.fn(async () => new Headers()),
  cookies: async () => ({
    getAll: () =>
      [...cookieJar.keys()].map((name) => ({
        name,
        value: cookieJar.get(name) ?? "",
      })),
    delete: (name: string) => {
      cookieJar.delete(name);
    },
  }),
}));

vi.mock("@sentry/nextjs", () => ({
  withServerActionInstrumentation: (
    _name: string,
    _options: object,
    fn: () => unknown,
  ) => fn(),
}));

import {
  getCurrentUser,
  getWebSocketToken,
  refreshSession,
  serverLogout,
  validateSession,
} from "../actions";

const sessionUser = {
  id: "user-1",
  email: "user@example.com",
  name: "Test User",
  role: "user",
  createdAt: new Date("2026-01-02T03:04:05.000Z"),
};

beforeEach(() => {
  getServerSessionMock.mockReset();
  getServerAuthTokenMock.mockReset();
  signOutMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("validateSession", () => {
  it("returns the mapped user and marks the session valid", async () => {
    getServerSessionMock.mockResolvedValue({ user: sessionUser });

    const result = await validateSession("/copilot");

    expect(result.isValid).toBe(true);
    expect(result.redirectPath).toBeUndefined();
    expect(result.user).toEqual({
      id: "user-1",
      email: "user@example.com",
      role: "authenticated",
      created_at: "2026-01-02T03:04:05.000Z",
      user_metadata: { name: "Test User", email: "user@example.com" },
    });
  });

  it("redirects protected paths to login when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const result = await validateSession("/copilot");

    expect(result.isValid).toBe(false);
    expect(result.user).toBeNull();
    expect(result.redirectPath).toBe(
      `/login?next=${encodeURIComponent("/copilot")}`,
    );
  });

  it("omits the redirect for public paths when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const result = await validateSession("/marketplace");

    expect(result.isValid).toBe(false);
    expect(result.redirectPath).toBeUndefined();
  });

  it("falls back to a login redirect when the session lookup throws", async () => {
    getServerSessionMock.mockRejectedValue(new Error("db unavailable"));

    const result = await validateSession("/library");

    expect(result.isValid).toBe(false);
    expect(result.redirectPath).toBe(
      `/login?next=${encodeURIComponent("/library")}`,
    );
  });
});

describe("getCurrentUser", () => {
  it("maps the session user", async () => {
    getServerSessionMock.mockResolvedValue({ user: sessionUser });

    const result = await getCurrentUser();

    expect(result.error).toBeUndefined();
    expect(result.user?.id).toBe("user-1");
    expect(result.user?.user_metadata).toEqual({
      name: "Test User",
      email: "user@example.com",
    });
  });

  it("returns a null user without an error when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const result = await getCurrentUser();

    expect(result).toEqual({ user: null });
  });

  it("surfaces the error message when the session lookup throws", async () => {
    getServerSessionMock.mockRejectedValue(new Error("session store down"));

    const result = await getCurrentUser();

    expect(result.user).toBeNull();
    expect(result.error).toBe("session store down");
  });
});

describe("getWebSocketToken", () => {
  it("returns the token from the server auth helper", async () => {
    getServerAuthTokenMock.mockResolvedValue("ws-token-123");

    const result = await getWebSocketToken();

    expect(result).toEqual({ token: "ws-token-123" });
  });

  it("returns a null token and the error message when token retrieval throws", async () => {
    getServerAuthTokenMock.mockRejectedValue(new Error("jwks unreachable"));

    const result = await getWebSocketToken();

    expect(result.token).toBeNull();
    expect(result.error).toBe("jwks unreachable");
  });
});

describe("serverLogout", () => {
  beforeEach(() => {
    cookieJar.clear();
  });

  it("signs out via Better Auth and clears the auth cookies", async () => {
    signOutMock.mockResolvedValue({ success: true });
    cookieJar.set("better-auth.session_token", "tok");
    cookieJar.set("better-auth.session_data", "data");
    cookieJar.set("sb-localhost-auth-token", "legacy");
    cookieJar.set("theme", "dark");

    const result = await serverLogout();

    expect(signOutMock).toHaveBeenCalledWith({
      headers: expect.any(Headers),
    });
    expect(result).toEqual({ success: true });
    // All auth cookies (Better Auth + leftover Supabase) are expired;
    // unrelated cookies survive.
    expect(cookieJar.has("better-auth.session_token")).toBe(false);
    expect(cookieJar.has("better-auth.session_data")).toBe(false);
    expect(cookieJar.has("sb-localhost-auth-token")).toBe(false);
    expect(cookieJar.has("theme")).toBe(true);
  });

  it("still clears the cookies and reports success when sign-out throws", async () => {
    // A failed server-side revocation must not leave a half-dead session
    // whose surviving cookie bounces the user back into the app.
    signOutMock.mockRejectedValue(new Error("session revocation failed"));
    cookieJar.set("better-auth.session_token", "tok");

    const result = await serverLogout();

    expect(result).toEqual({ success: true });
    expect(cookieJar.has("better-auth.session_token")).toBe(false);
  });
});

describe("refreshSession", () => {
  it("returns the mapped user when the session is still active", async () => {
    getServerSessionMock.mockResolvedValue({ user: sessionUser });

    const result = await refreshSession();

    expect(result).toEqual({
      user: {
        id: "user-1",
        email: "user@example.com",
        role: "authenticated",
        created_at: "2026-01-02T03:04:05.000Z",
        user_metadata: { name: "Test User", email: "user@example.com" },
      },
    });
  });

  it("returns an error when there is no active session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const result = await refreshSession();

    expect(result).toEqual({ user: null, error: "No active session" });
  });

  it("returns the error message when the session lookup throws", async () => {
    getServerSessionMock.mockRejectedValue(new Error("connection reset"));

    const result = await refreshSession();

    expect(result).toEqual({ user: null, error: "connection reset" });
  });
});
