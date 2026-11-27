// @vitest-environment node
// The browser environment's Response drops Set-Cookie, which carries the
// cookie cache from the server to the visitor.
import { createHmac, randomUUID } from "node:crypto";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { toNodeHandler } from "better-auth/node";
import { http, passthrough } from "msw";
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { server as mockServer } from "@/mocks/mock-server";
import { authMiddleware } from "../middleware";

// The middleware's admin check against the real Better Auth server from
// auth.ts (a memory database, its rate limits on as in production), with the
// cookies that server sets, over http and over https (#17308).

const testDB = vi.hoisted(() => ({ current: {} as Record<string, unknown[]> }));
const buildMode = vi.hoisted(() => ({ production: true }));

vi.mock("pg", () => ({
  Pool: class {
    async query(sql: string) {
      throw new Error(`unexpected query: ${sql}`);
    }
  },
}));
vi.mock("better-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth")>();
  const { memoryAdapter } = await import("better-auth/adapters/memory");
  return {
    ...actual,
    betterAuth: (options: Parameters<typeof actual.betterAuth>[0]) =>
      actual.betterAuth({
        ...options,
        database: memoryAdapter(testDB.current),
        // The admin plugin gives a user a role; the rest need tables this
        // database doesn't have.
        plugins: options.plugins?.filter((plugin) => plugin.id === "admin"),
        rateLimit: { ...options.rateLimit, enabled: true },
      }),
  };
});
// Better Auth sets getCookieCache's default cookie name when it loads, from
// NODE_ENV: the `__Secure-` one in a production build. Tests run with
// NODE_ENV=test, so this puts back a production (or development) build's.
vi.mock("better-auth/cookies", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth/cookies")>();
  return {
    ...actual,
    getCookieCache: (
      request: Parameters<typeof actual.getCookieCache>[0],
      config?: Parameters<typeof actual.getCookieCache>[1],
    ) =>
      actual.getCookieCache(request, {
        ...config,
        isSecure: config?.isSecure ?? buildMode.production,
      }),
  };
});
vi.mock("better-auth/next-js", () => ({
  nextCookies: vi.fn(() => ({ id: "next-cookies" })),
}));

const SECRET = "test-secret-that-is-at-least-32-chars"; // pragma: allowlist secret
const OTHER_SECRET = "another-secret-at-least-32-chars-long"; // pragma: allowlist secret
const DAY_MS = 24 * 60 * 60 * 1000;

interface AuthServer {
  baseURL: string;
  handler: (request: Request) => Promise<Response>;
  sessionCookieName: string;
  rateLimitMax: number;
}

async function startAuthServer(baseURL: string): Promise<AuthServer> {
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", SECRET);
  vi.doUnmock("../auth");
  vi.resetModules();
  testDB.current = {
    UserAuthIdentity: [],
    UserAuthSession: [],
    UserAuthAccount: [],
    UserAuthVerification: [],
  };
  const { auth } = await import("../auth");
  const context = await auth.$context;
  return {
    baseURL,
    handler: auth.handler,
    sessionCookieName: context.authCookies.sessionToken.name,
    rateLimitMax: context.rateLimit.max,
  };
}

// A session cookie as Better Auth signs it: the token, a dot and the token's
// base64 HMAC-SHA256, URL-encoded.
function sessionCookie(server: AuthServer, token: string, secret = SECRET) {
  const signature = createHmac("sha256", secret).update(token).digest("base64");
  return `${server.sessionCookieName}=${encodeURIComponent(`${token}.${signature}`)}`;
}

// The signed session cookie, and the cookie cache the server sets with it.
async function signIn(server: AuthServer, role: "admin" | "user") {
  const now = new Date();
  const userID = randomUUID();
  const token = randomUUID().replaceAll("-", "");
  testDB.current.UserAuthIdentity.push({
    id: userID,
    email: `${role}-${userID}@example.com`,
    name: role,
    emailVerified: true,
    role,
    createdAt: now,
    updatedAt: now,
  });
  testDB.current.UserAuthSession.push({
    id: randomUUID(),
    token,
    userId: userID,
    expiresAt: new Date(now.getTime() + 30 * DAY_MS),
    createdAt: now,
    updatedAt: now,
  });
  const signedCookie = sessionCookie(server, token);
  const response = await server.handler(
    new Request(`${server.baseURL}/api/auth/get-session`, {
      headers: { cookie: signedCookie, "x-forwarded-for": nextIP() },
    }),
  );
  const cacheCookies = response.headers
    .getSetCookie()
    .map((cookie) => cookie.split(";")[0]);
  expect(response.status).toBe(200);
  expect(cacheCookies).toHaveLength(1);
  return { token, sessionCookie: signedCookie, cacheCookie: cacheCookies[0] };
}

// The same signed cache, but claiming the admin role.
function claimingAdmin(cacheCookie: string) {
  const separator = cacheCookie.indexOf("=");
  const value = cacheCookie.slice(separator + 1);
  const cache = JSON.parse(Buffer.from(value, "base64url").toString());
  cache.session.user.role = "admin";
  const forged = Buffer.from(JSON.stringify(cache)).toString("base64url");
  return `${cacheCookie.slice(0, separator)}=${forged}`;
}

let addresses = 0;
// Every sign-in and visitor comes from an address of its own, so no test
// spends another's rate limit.
// Documentation addresses (RFC 5737), each test's own rate-limit bucket.
const TEST_NETS = ["198.51.100", "203.0.113", "192.0.2"];

function nextIP() {
  const net = TEST_NETS[Math.floor(addresses / 254)];
  if (!net) throw new Error("the tests used up the documentation addresses");
  const ip = `${net}.${(addresses % 254) + 1}`;
  addresses += 1;
  return ip;
}

function adminPage(server: AuthServer, cookies: string[], ip: string) {
  return new NextRequest(`${server.baseURL}/admin/users`, {
    headers: { cookie: cookies.join("; "), "x-forwarded-for": ip },
  });
}

function redirectPath(response: Response) {
  const location = response.headers.get("location");
  return location === null ? null : new URL(location).pathname;
}

describe.each([
  { scheme: "http", baseURL: "http://localhost:3000" },
  { scheme: "https", baseURL: "https://platform.example.com" },
])("the admin check over $scheme", ({ baseURL }) => {
  let server: AuthServer;
  let lookups: Request[];

  beforeEach(async () => {
    buildMode.production = true;
    server = await startAuthServer(baseURL);
    lookups = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input, init) => {
      const request = new Request(input, init);
      lookups.push(request);
      return new URL(request.url).pathname === "/api/auth/get-session"
        ? server.handler(request)
        : new Response("not the session endpoint", { status: 404 });
    });
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  // Better Auth's default cookie name follows the build mode; the
  // middleware's must follow the base URL in both.
  it.each(["production", "development"])(
    "lets an admin through on their own session's cookie cache, with no session lookup, in a %s build",
    async (mode) => {
      buildMode.production = mode === "production";
      const admin = await signIn(server, "admin");
      const ip = nextIP();

      // Each admin page load checks the page and prefetches every admin link:
      // more checks in a row than the session endpoint answers.
      const bounced: number[] = [];
      for (let check = 0; check <= server.rateLimitMax; check++) {
        const response = await authMiddleware(
          adminPage(server, [admin.sessionCookie, admin.cacheCookie], ip),
        );
        if (redirectPath(response) !== null) bounced.push(check);
      }

      expect(bounced).toEqual([]);
      expect(lookups).toHaveLength(0);
    },
  );

  it("sends a non-admin to Home", async () => {
    const member = await signIn(server, "user");

    const response = await authMiddleware(
      adminPage(server, [member.sessionCookie, member.cacheCookie], nextIP()),
    );

    expect(redirectPath(response)).toBe("/");
  });

  it("sends a non-admin whose cookie cache is altered to claim admin to Home", async () => {
    const member = await signIn(server, "user");

    const response = await authMiddleware(
      adminPage(
        server,
        [member.sessionCookie, claimingAdmin(member.cacheCookie)],
        nextIP(),
      ),
    );

    expect(redirectPath(response)).toBe("/");
  });

  it("refuses a cookie cache that belongs to another session", async () => {
    const admin = await signIn(server, "admin");
    const member = await signIn(server, "user");

    const response = await authMiddleware(
      adminPage(server, [member.sessionCookie, admin.cacheCookie], nextIP()),
    );

    // The session lookup decides instead, and finds the member.
    expect(redirectPath(response)).toBe("/");
    expect(lookups).toHaveLength(1);
  });

  it("goes by the first of two sessions' cookies, as the session lookup does", async () => {
    const admin = await signIn(server, "admin");
    const member = await signIn(server, "user");
    const cookies = [
      member.sessionCookie,
      member.cacheCookie,
      admin.sessionCookie,
      admin.cacheCookie,
    ];

    const response = await authMiddleware(adminPage(server, cookies, nextIP()));
    const lookup = await server.handler(
      new Request(`${server.baseURL}/api/auth/get-session`, {
        headers: { cookie: cookies.join("; "), "x-forwarded-for": nextIP() },
      }),
    );

    expect((await lookup.json()).user.role).toBe("user");
    expect(redirectPath(response)).toBe("/");
  });

  it.each<{
    session: string;
    cookie: (server: AuthServer, cachedToken: string) => string;
  }>([
    {
      session: "a session token the server never issued",
      cookie: (server) => `${server.sessionCookieName}=made-up`,
    },
    {
      session: "the cached session's token without its signature",
      cookie: (server, cachedToken) =>
        `${server.sessionCookieName}=${cachedToken}`,
    },
    {
      session: "the cached session's token signed with another secret",
      cookie: (server, cachedToken) =>
        sessionCookie(server, cachedToken, OTHER_SECRET),
    },
    {
      // Well formed and signed with the right secret, but over another
      // token, so the signature comparison itself refuses it.
      session: "the cached session's token with another token's signature",
      cookie: (server, cachedToken) => {
        const signature = createHmac("sha256", SECRET)
          .update(`${cachedToken}-other`)
          .digest("base64");
        return `${server.sessionCookieName}=${encodeURIComponent(`${cachedToken}.${signature}`)}`;
      },
    },
  ])("refuses a cookie cache alongside $session", async ({ cookie }) => {
    const admin = await signIn(server, "admin");

    const response = await authMiddleware(
      adminPage(
        server,
        [cookie(server, admin.token), admin.cacheCookie],
        nextIP(),
      ),
    );

    expect(redirectPath(response)).toBe("/");
  });

  it("sends an expired admin session to Home", async () => {
    const admin = await signIn(server, "admin");
    vi.useFakeTimers({ toFake: ["Date"], now: Date.now() + 31 * DAY_MS });

    const response = await authMiddleware(
      adminPage(server, [admin.sessionCookie, admin.cacheCookie], nextIP()),
    );

    expect(redirectPath(response)).toBe("/");
  });

  it("sends a signed-out visitor, even one holding an admin's cookie cache, to sign in", async () => {
    const admin = await signIn(server, "admin");

    for (const cookies of [[], [admin.cacheCookie]]) {
      const response = await authMiddleware(
        adminPage(server, cookies, nextIP()),
      );
      expect(redirectPath(response)).toBe("/login");
    }
  });

  it("counts the session lookups made for a lapsed cookie cache against each visitor", async () => {
    // The cache lasts five minutes. Until a session check sets it again,
    // every admin check looks the session up.
    const first = await signIn(server, "admin");
    const second = await signIn(server, "admin");
    const firstIP = nextIP();

    const bounced: number[] = [];
    for (let check = 0; check < server.rateLimitMax; check++) {
      const response = await authMiddleware(
        adminPage(server, [first.sessionCookie], firstIP),
      );
      if (redirectPath(response) !== null) bounced.push(check);
    }
    const secondVisit = await authMiddleware(
      adminPage(server, [second.sessionCookie], nextIP()),
    );

    expect(bounced).toEqual([]);
    expect(redirectPath(secondVisit)).toBeNull();
  });

  it.each([
    {
      through: "a proxy chain",
      forwardedFor: () => `${nextIP()}, ${nextIP()}`,
    },
    { through: "no proxy", forwardedFor: () => null },
  ])(
    "gives a session lookup through $through only the visitor's cookies and X-Forwarded-For, as they arrived",
    async ({ forwardedFor }) => {
      const admin = await signIn(server, "admin");
      const visitor = forwardedFor();
      const request = new NextRequest(`${server.baseURL}/admin/users`, {
        headers: {
          cookie: admin.sessionCookie,
          "x-real-ip": nextIP(),
          authorization: "Bearer someone-else",
          ...(visitor === null ? {} : { "x-forwarded-for": visitor }),
        },
      });

      const response = await authMiddleware(request);

      expect(redirectPath(response)).toBeNull();
      expect(lookups.map((lookup) => new URL(lookup.url).pathname)).toEqual([
        "/api/auth/get-session",
      ]);
      expect(Object.fromEntries(lookups[0].headers)).toEqual({
        cookie: admin.sessionCookie,
        ...(visitor === null ? {} : { "x-forwarded-for": visitor }),
      });
    },
  );
});

// The server on a port of its own, over plain http, as Next serves behind a
// proxy that ends TLS (#17339). There Next builds request.url from its own
// address and the proxy's X-Forwarded-Proto: https, which that port doesn't
// speak. The public URL is https, so the cookies carry `__Secure-` names.
describe("the admin check behind a proxy that ends TLS", () => {
  let server: AuthServer;
  let listener: Server;
  let port: number;
  let arrived: string[];

  beforeEach(async () => {
    buildMode.production = true;
    // The lookup's three-second deadline is the product's. A runner that
    // pauses past it would send the admin Home on correct code, so these
    // tests own the lookup's signal; the timeout has its own test below.
    vi.spyOn(AbortSignal, "timeout").mockImplementation(
      () => new AbortController().signal,
    );
    server = await startAuthServer("https://platform.example.com");
    arrived = [];
    const handle = toNodeHandler(server.handler);
    listener = createServer((request, response) => {
      arrived.push(request.url ?? "");
      void handle(request, response);
    });
    await new Promise<void>((resolve) =>
      listener.listen(0, "127.0.0.1", resolve),
    );
    port = (listener.address() as AddressInfo).port;
    // NextRequest names a loopback address `localhost`, as Next does.
    mockServer.use(
      ...["http", "https"].flatMap((scheme) =>
        ["127.0.0.1", "localhost"].map((host) =>
          http.get(`${scheme}://${host}:${port}/api/auth/get-session`, () =>
            passthrough(),
          ),
        ),
      ),
    );
  });

  afterEach(async () => {
    listener.closeAllConnections();
    await new Promise((resolve) => listener.close(resolve));
    vi.unstubAllEnvs();
    vi.restoreAllMocks();
  });

  // Only the session cookie: the five-minute cookie cache has lapsed.
  function adminPage(sessionCookie: string) {
    return new NextRequest(`https://localhost:${port}/admin/users`, {
      headers: { cookie: sessionCookie, "x-forwarded-for": nextIP() },
    });
  }

  it("lets an admin through by looking their session up at BETTER_AUTH_INTERNAL_URL", async () => {
    vi.stubEnv("BETTER_AUTH_INTERNAL_URL", `http://127.0.0.1:${port}`);
    const admin = await signIn(server, "admin");
    const request = adminPage(admin.sessionCookie);

    // The request's own address is https on a plain-http port, so a lookup
    // there fails. How it fails depends on the TLS library and timing.
    await expect(
      fetch(new URL("/api/auth/get-session", request.url)),
    ).rejects.toThrow();
    const response = await authMiddleware(request);

    expect(redirectPath(response)).toBeNull();
    expect(arrived).toEqual(["/api/auth/get-session"]);
  });

  it("sends a non-admin Home", async () => {
    vi.stubEnv("BETTER_AUTH_INTERNAL_URL", `http://127.0.0.1:${port}`);
    const member = await signIn(server, "user");

    const response = await authMiddleware(adminPage(member.sessionCookie));

    expect(redirectPath(response)).toBe("/");
    expect(arrived).toEqual(["/api/auth/get-session"]);
  });

  // A lookup that runs out its deadline counts as "not an admin", the safe
  // default, rather than holding the page.
  it("sends an admin Home when the session lookup times out", async () => {
    vi.stubEnv("BETTER_AUTH_INTERNAL_URL", `http://127.0.0.1:${port}`);
    const admin = await signIn(server, "admin");
    vi.spyOn(AbortSignal, "timeout").mockImplementation(() =>
      AbortSignal.abort(
        new DOMException(
          "The operation was aborted due to timeout",
          "TimeoutError",
        ),
      ),
    );

    const response = await authMiddleware(adminPage(admin.sessionCookie));

    expect(redirectPath(response)).toBe("/");
    expect(vi.mocked(AbortSignal.timeout)).toHaveBeenCalled();
  });
});
