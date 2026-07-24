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

// Fresh module per test to reset the module-scope token cache. React's cache()
// wrapper is shimmed to identity in the global vitest setup, so per-request
// memoization is inert here.
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

  it("mints a token via auth.api.getToken and serves repeats from the cache", async () => {
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
    expect(getTokenMock).toHaveBeenCalledTimes(1);
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

  it("returns null instead of throwing when the token mint fails", async () => {
    cookiesMock.mockResolvedValue(
      cookieStore([
        { name: "better-auth.session_token", value: "session-error" },
      ]),
    );
    getTokenMock.mockRejectedValue(new Error("jwks unavailable"));
    const { getServerAuthToken } = await importModule();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });
});

describe("server token cache", () => {
  it("reads the exp claim from a JWT", async () => {
    const { readJwtExpiryMs } = await importModule();
    const token = makeJwt(3600);

    const expiry = readJwtExpiryMs(token);

    expect(expiry).toBeGreaterThan(Date.now() + 3500 * 1000);
    expect(expiry).toBeLessThan(Date.now() + 3700 * 1000);
  });

  it("falls back to a conservative default for malformed tokens", async () => {
    const { readJwtExpiryMs } = await importModule();

    const expiry = readJwtExpiryMs("not-a-jwt");

    expect(expiry).toBeGreaterThan(Date.now());
  });

  it("returns a cached token until its expiry margin", async () => {
    const { cacheServerToken, getCachedServerToken } = await importModule();
    const token = makeJwt(3600);
    cacheServerToken("session-cookie-1", token);

    expect(getCachedServerToken("session-cookie-1")).toBe(token);
    expect(getCachedServerToken("other-session")).toBeNull();
  });

  it("does not return tokens that are within the expiry margin", async () => {
    const { cacheServerToken, getCachedServerToken } = await importModule();
    // 60s to expiry is inside the 5-minute refresh margin
    cacheServerToken("session-cookie-2", makeJwt(60));

    expect(getCachedServerToken("session-cookie-2")).toBeNull();
  });
});
