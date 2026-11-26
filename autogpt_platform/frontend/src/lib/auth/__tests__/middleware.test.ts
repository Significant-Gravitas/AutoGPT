import { createHmac } from "node:crypto";
import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getSessionCookie = vi.fn();
const getCookieCache = vi.fn();

vi.mock("better-auth/cookies", async (importOriginal) => ({
  ...(await importOriginal<typeof import("better-auth/cookies")>()),
  getSessionCookie: (...args: unknown[]) => getSessionCookie(...args),
  getCookieCache: (...args: unknown[]) => getCookieCache(...args),
}));

import { authMiddleware } from "../middleware";

const SECRET = "test-secret-that-is-at-least-32-chars"; // pragma: allowlist secret

function makeRequest(path: string, cookies: Record<string, string> = {}) {
  const request = new NextRequest(new URL(`http://localhost:3000${path}`));
  for (const [name, value] of Object.entries(cookies)) {
    request.cookies.set(name, value);
  }
  return request;
}

// The session cookie, signed as Better Auth signs it, and a cookie cache of
// that session for a user with `role`.
function signedInAs(role: string) {
  vi.stubEnv("BETTER_AUTH_URL", "http://localhost:3000");
  vi.stubEnv("BETTER_AUTH_SECRET", SECRET);
  const token = "session-token";
  const signature = createHmac("sha256", SECRET).update(token).digest("base64");
  getSessionCookie.mockReturnValue(`${token}.${signature}`);
  getCookieCache.mockResolvedValue({ session: { token }, user: { role } });
  return { "better-auth.session_token": `${token}.${signature}` };
}

function refuseSessionLookups() {
  return vi
    .spyOn(globalThis, "fetch")
    .mockRejectedValue(new Error("unexpected session lookup"));
}

beforeEach(() => {
  getSessionCookie.mockReset();
  getCookieCache.mockReset();
  getCookieCache.mockResolvedValue(null);
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("authMiddleware", () => {
  it("passes through API routes without auth checks", async () => {
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(makeRequest("/api/auth/get-session"));

    expect(response.headers.get("location")).toBeNull();
    expect(getSessionCookie).not.toHaveBeenCalled();
  });

  it("redirects unauthenticated users on protected pages to login", async () => {
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(makeRequest("/copilot?foo=bar"));

    const location = response.headers.get("location");
    expect(location).toContain("/login");
    expect(location).toContain(encodeURIComponent("/copilot?foo=bar"));
  });

  it("lets unauthenticated users browse public pages", async () => {
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(makeRequest("/marketplace"));

    expect(response.headers.get("location")).toBeNull();
  });

  it("redirects legacy Supabase sessions to the bridge endpoint", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", "legacy-secret");
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(
      makeRequest("/copilot", { "sb-proj-auth-token": "legacy" }),
    );

    const location = response.headers.get("location");
    expect(location).toContain("/api/auth/supabase-bridge");
    expect(location).toContain(encodeURIComponent("/copilot"));
  });

  it("sends legacy-cookie users to login, not the bridge, when the legacy secret is unset", async () => {
    // The bridge can't consume cookies without SUPABASE_JWT_SECRET and
    // deliberately leaves them intact — so bouncing there would loop
    // middleware -> bridge -> /login -> middleware forever.
    vi.stubEnv("SUPABASE_JWT_SECRET", "");
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(
      makeRequest("/copilot", { "sb-proj-auth-token": "legacy" }),
    );

    const location = response.headers.get("location");
    expect(location).toContain("/login");
    expect(location).not.toContain("supabase-bridge");
  });

  it("lets legacy-cookie users browse public pages when the legacy secret is unset", async () => {
    vi.stubEnv("SUPABASE_JWT_SECRET", "");
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(
      makeRequest("/marketplace", { "sb-proj-auth-token": "legacy" }),
    );

    expect(response.headers.get("location")).toBeNull();
  });

  it("does not bridge when a Better Auth session already exists", async () => {
    getSessionCookie.mockReturnValue("session-token");

    const response = await authMiddleware(
      makeRequest("/copilot", { "sb-proj-auth-token": "legacy" }),
    );

    expect(response.headers.get("location")).toBeNull();
  });

  it("redirects non-admin users away from admin pages", async () => {
    const cookies = signedInAs("user");
    const lookups = refuseSessionLookups();

    const response = await authMiddleware(makeRequest("/admin", cookies));

    expect(response.headers.get("location")).toBe("http://localhost:3000/");
    expect(lookups).not.toHaveBeenCalled();
  });

  it("lets admins through to admin pages via their session's cookie cache", async () => {
    const cookies = signedInAs("admin");
    const lookups = refuseSessionLookups();

    const response = await authMiddleware(makeRequest("/admin", cookies));

    expect(response.headers.get("location")).toBeNull();
    expect(lookups).not.toHaveBeenCalled();
  });

  it("falls back to the session endpoint when the cookie cache is empty", async () => {
    getSessionCookie.mockReturnValue("session-token");
    getCookieCache.mockResolvedValue(null);
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ user: { role: "admin" } }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const response = await authMiddleware(makeRequest("/admin"));

    expect(fetchSpy).toHaveBeenCalled();
    expect(response.headers.get("location")).toBeNull();
    fetchSpy.mockRestore();
  });

  describe("the session lookup's address", () => {
    function answerAsAdmin() {
      return vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ user: { role: "admin" } }), {
          status: 200,
          headers: { "content-type": "application/json" },
        }),
      );
    }

    function lookedUpAt(lookups: ReturnType<typeof answerAsAdmin>) {
      expect(lookups).toHaveBeenCalledOnce();
      return String(lookups.mock.calls[0][0]);
    }

    it("is BETTER_AUTH_INTERNAL_URL behind a proxy that ends TLS", async () => {
      vi.stubEnv("BETTER_AUTH_INTERNAL_URL", "http://127.0.0.1:3001");
      getSessionCookie.mockReturnValue("session-token");
      const lookups = answerAsAdmin();

      // What Next builds there: its own address, with the proxy's
      // X-Forwarded-Proto.
      const response = await authMiddleware(
        new NextRequest("https://127.0.0.1:3001/admin"),
      );

      expect(lookedUpAt(lookups)).toBe(
        "http://127.0.0.1:3001/api/auth/get-session",
      );
      expect(response.headers.get("location")).toBeNull();
    });

    it.each([
      ["unset", undefined],
      ["empty", ""],
    ])(
      "is the request's own origin when BETTER_AUTH_INTERNAL_URL is %s",
      async (_how, internalURL) => {
        vi.stubEnv("BETTER_AUTH_INTERNAL_URL", internalURL);
        getSessionCookie.mockReturnValue("session-token");
        const lookups = answerAsAdmin();

        const response = await authMiddleware(makeRequest("/admin"));

        expect(lookedUpAt(lookups)).toBe(
          "http://localhost:3000/api/auth/get-session",
        );
        expect(response.headers.get("location")).toBeNull();
      },
    );
  });

  it.each([
    ["is turned down for too many requests", 429],
    ["fails on the server", 500],
  ])(
    "sends the visitor Home when the session lookup %s",
    async (_how, status) => {
      getSessionCookie.mockReturnValue("session-token");
      // Whatever the body says: only an answered lookup names the visitor.
      vi.spyOn(globalThis, "fetch").mockResolvedValue(
        new Response(JSON.stringify({ user: { role: "admin" } }), {
          status,
          headers: { "content-type": "application/json" },
        }),
      );

      const response = await authMiddleware(makeRequest("/admin"));

      expect(response.headers.get("location")).toBe("http://localhost:3000/");
    },
  );

  it("sends the visitor Home when the session lookup cannot be made", async () => {
    getSessionCookie.mockReturnValue("session-token");
    const lookups = refuseSessionLookups();

    const response = await authMiddleware(makeRequest("/admin"));

    expect(lookups).toHaveBeenCalledOnce();
    expect(response.headers.get("location")).toBe("http://localhost:3000/");
  });

  it("sends the visitor Home when the session lookup answers with no session", async () => {
    getSessionCookie.mockReturnValue("session-token");
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("null", {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const response = await authMiddleware(makeRequest("/admin"));

    expect(response.headers.get("location")).toBe("http://localhost:3000/");
  });
});
