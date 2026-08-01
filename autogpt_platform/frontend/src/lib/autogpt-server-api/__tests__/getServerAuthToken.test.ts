import { createRequire } from "node:module";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const isClientSideMock = vi.fn(() => false);
const getAllCookiesMock = vi.fn<() => { name: string; value: string }[]>(
  () => [],
);

vi.mock("@/services/environment", () => ({
  environment: {
    isClientSide: () => isClientSideMock(),
    isServerSide: () => !isClientSideMock(),
    getAGPTServerApiUrl: () => "http://localhost:8006/api",
  },
}));

// getServerAuthToken reads cookies through a lazy `require("next/headers")`,
// which resolves via Node's CJS loader and therefore bypasses vi.mock. Spy on
// the shared CJS module instance instead so the production require sees the
// stubbed cookie store.
const cjsRequire = createRequire(import.meta.url);
const nextHeadersCjs = cjsRequire(
  "next/headers",
) as typeof import("next/headers");
type CookieStore = Awaited<ReturnType<typeof nextHeadersCjs.cookies>>;

const fetchMock = vi.fn();

function makeJwt(expSecondsFromNow: number): string {
  const payload = Buffer.from(
    JSON.stringify({ exp: Math.floor(Date.now() / 1000) + expSecondsFromNow }),
  ).toString("base64url");
  return `header.${payload}.signature`;
}

// Each test imports a fresh copy of the module to stay isolated. React's
// cache() wrapper is shimmed to identity in the global vitest setup, so
// per-request memoization is inert here — every call really does re-mint.
async function importGetServerAuthToken() {
  vi.resetModules();
  const helpers = await import("../helpers");
  return helpers.getServerAuthToken;
}

beforeEach(() => {
  isClientSideMock.mockReset().mockReturnValue(false);
  getAllCookiesMock.mockReset().mockReturnValue([]);
  fetchMock.mockReset();
  vi.spyOn(nextHeadersCjs, "cookies").mockImplementation(
    async () => ({ getAll: () => getAllCookiesMock() }) as CookieStore,
  );
  vi.stubGlobal("fetch", fetchMock);
  vi.stubEnv("BETTER_AUTH_URL", "http://auth.internal:3000");
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("getServerAuthToken", () => {
  it("returns null on the client side without reading cookies or fetching", async () => {
    isClientSideMock.mockReturnValue(true);
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
    expect(getAllCookiesMock).not.toHaveBeenCalled();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("returns null without fetching when there is no Better Auth session cookie", async () => {
    getAllCookiesMock.mockReturnValue([
      { name: "unrelated-cookie", value: "abc" },
    ]);
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("re-mints on every call rather than caching across requests", async () => {
    // A cross-request cache would return a token without re-checking the
    // session, outliving logout / password-reset revocation. Going back to
    // /api/auth/token each time is what re-validates it.
    const jwt = makeJwt(3600);
    getAllCookiesMock.mockReturnValue([
      { name: "better-auth.session_token", value: "session-fresh" },
      { name: "other", value: "with spaces" },
    ]);
    // A fresh Response per call — a body can only be consumed once, so a shared
    // instance would make the second call look like a failure.
    fetchMock.mockImplementation(
      async () => new Response(JSON.stringify({ token: jwt }), { status: 200 }),
    );
    const getServerAuthToken = await importGetServerAuthToken();

    const first = await getServerAuthToken();
    const second = await getServerAuthToken();

    expect(first).toBe(jwt);
    expect(second).toBe(jwt);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(fetchMock).toHaveBeenCalledWith(
      "http://auth.internal:3000/api/auth/token",
      {
        headers: {
          cookie:
            "better-auth.session_token=session-fresh; other=with%20spaces",
        },
        cache: "no-store",
        // Bounded on purpose: this is a request from the Next server to itself,
        // so an unbounded wait can deadlock a worker instead of erroring.
        signal: expect.any(AbortSignal),
      },
    );
  });

  it("recognizes the __Secure-prefixed session cookie", async () => {
    const jwt = makeJwt(3600);
    getAllCookiesMock.mockReturnValue([
      { name: "__Secure-better-auth.session_token", value: "session-secure" },
    ]);
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ token: jwt }), { status: 200 }),
    );
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBe(jwt);
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("returns null when the token endpoint responds with a non-200 status", async () => {
    getAllCookiesMock.mockReturnValue([
      { name: "better-auth.session_token", value: "session-unauthorized" },
    ]);
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({ error: "unauthorized" }), { status: 401 }),
    );
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });

  it("returns null when the token endpoint responds without a token field", async () => {
    getAllCookiesMock.mockReturnValue([
      { name: "better-auth.session_token", value: "session-empty-body" },
    ]);
    fetchMock.mockResolvedValue(
      new Response(JSON.stringify({}), { status: 200 }),
    );
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });

  it("returns null instead of throwing when the token fetch fails", async () => {
    getAllCookiesMock.mockReturnValue([
      { name: "better-auth.session_token", value: "session-network-error" },
    ]);
    fetchMock.mockRejectedValue(new TypeError("fetch failed"));
    const getServerAuthToken = await importGetServerAuthToken();

    const token = await getServerAuthToken();

    expect(token).toBeNull();
  });
});
