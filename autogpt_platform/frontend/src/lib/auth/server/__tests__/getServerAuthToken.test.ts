import { APIError } from "better-auth/api";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getTokenMock = vi.fn();
const cookiesMock = vi.fn();
const headersMock = vi.fn();

// Replaces the real auth instance so the module never pulls in `pg`, and so we
// can assert the in-process token mint instead of the old HTTP self-fetch.
vi.mock("@/lib/auth/auth", () => ({
  auth: {
    api: {
      getToken: (...args: unknown[]) => getTokenMock(...args),
    },
  },
}));

vi.mock("next/headers", () => ({
  cookies: () => cookiesMock(),
  headers: () => headersMock(),
}));

function makeJwt(expSecondsFromNow: number): string {
  const payload = Buffer.from(
    JSON.stringify({ exp: Math.floor(Date.now() / 1000) + expSecondsFromNow }),
  ).toString("base64url");
  return `header.${payload}.signature`;
}

function cookieStore(cookies: { name: string; value: string }[]) {
  return { getAll: () => cookies };
}

// Fresh module per test for isolation. React's cache() wrapper is shimmed to
// identity in the global vitest setup, so per-request memoization is inert
// here — every call really does re-mint.
async function importModule() {
  vi.resetModules();
  return await import("../getServerAuthToken");
}

beforeEach(() => {
  getTokenMock.mockReset();
  cookiesMock.mockReset().mockResolvedValue(cookieStore([]));
  headersMock.mockReset().mockResolvedValue(new Headers());
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getServerAuthToken", () => {
  it("returns null without minting a token when there is no Better Auth session cookie", async () => {
    cookiesMock.mockResolvedValue(
      cookieStore([{ name: "unrelated-cookie", value: "abc" }]),
    );
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
    expect(getTokenMock).not.toHaveBeenCalled();
  });

  it("re-mints on every call rather than caching across requests", async () => {
    // A cross-request cache keyed on the session cookie would hand back a token
    // without re-checking the session, so a stolen cookie would keep working
    // after logout or revokeSessionsOnPasswordReset deleted the session row —
    // the backend only verifies the signature, never session existence.
    // Re-minting is what makes revocation actually take effect.
    const jwt = makeJwt(3600);
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "better-auth.session_token", value: "session-fresh" },
      ]),
    );
    getTokenMock.mockResolvedValue({ token: jwt });
    const { getServerAuthToken } = await importModule();

    const first = await getServerAuthToken();
    const second = await getServerAuthToken();

    expect(first).toBe(jwt);
    expect(second).toBe(jwt);
    expect(getTokenMock).toHaveBeenCalledTimes(2);
    expect(getTokenMock).toHaveBeenCalledWith({ headers: expect.any(Headers) });
  });

  it("recognizes the __Secure-prefixed session cookie", async () => {
    const jwt = makeJwt(3600);
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "__Secure-better-auth.session_token", value: "session-secure" },
      ]),
    );
    getTokenMock.mockResolvedValue({ token: jwt });
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBe(jwt);
    expect(getTokenMock).toHaveBeenCalledTimes(1);
  });

  it("returns null when auth.api.getToken yields no token", async () => {
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "better-auth.session_token", value: "session-empty" },
      ]),
    );
    getTokenMock.mockResolvedValue({ token: undefined });
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });

  it("returns null outside a request, where there are no cookies to read", async () => {
    cookiesMock.mockRejectedValue(
      new Error("cookies() was called outside a request scope"),
    );
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
    expect(getTokenMock).not.toHaveBeenCalled();
  });

  it("returns null when Better Auth rejects the session", async () => {
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "better-auth.session_token", value: "session-expired" },
      ]),
    );
    getTokenMock.mockRejectedValue(
      APIError.from("UNAUTHORIZED", {
        message: "Unauthorized",
        code: "UNAUTHORIZED",
      }),
    );
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });

  it("throws a retryable 503 when a live session's token cannot be minted", async () => {
    // What the appliance logged when its frontend role ran out of
    // connections: the signing-key read failed after the session checked out.
    // Returning null sent the request to the backend without a token.
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "better-auth.session_token", value: "session-error" },
      ]),
    );
    getTokenMock.mockRejectedValue(
      Object.assign(
        new Error('too many connections for role "autogpt_frontend"'),
        { code: "53300" },
      ),
    );
    const { getServerAuthToken } = await importModule();

    await expect(getServerAuthToken()).rejects.toMatchObject({
      name: "AuthTokenUnavailableError",
      status: 503,
      response: { code: "auth_token_unavailable" },
    });
  });
});
