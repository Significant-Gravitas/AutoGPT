import { NextRequest } from "next/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getSessionCookie = vi.fn();
const getCookieCache = vi.fn();

vi.mock("better-auth/cookies", () => ({
  getSessionCookie: (...args: unknown[]) => getSessionCookie(...args),
  getCookieCache: (...args: unknown[]) => getCookieCache(...args),
}));

import { authMiddleware } from "../middleware";

function makeRequest(path: string, cookies: Record<string, string> = {}) {
  const request = new NextRequest(new URL(`http://localhost:3000${path}`));
  for (const [name, value] of Object.entries(cookies)) {
    request.cookies.set(name, value);
  }
  return request;
}

beforeEach(() => {
  getSessionCookie.mockReset();
  getCookieCache.mockReset();
  getCookieCache.mockResolvedValue(null);
});

afterEach(() => {
  vi.unstubAllEnvs();
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
    getSessionCookie.mockReturnValue("session-token");
    getCookieCache.mockResolvedValue({ user: { role: "user" } });

    const response = await authMiddleware(makeRequest("/admin"));

    expect(response.headers.get("location")).toBe("http://localhost:3000/");
  });

  it("verifies cached admins against the DB so revoked admin sessions lose access", async () => {
    // REL-001: the cookie cache is a hint. A cached admin role must be
    // confirmed by the session endpoint — a revoked/demoted admin cannot
    // keep admin access via a stale signed cookie.
    getSessionCookie.mockReturnValue("session-token");
    getCookieCache.mockResolvedValue({ user: { role: "admin" } });
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

  it("drops a cached admin to a non-admin DB verdict", async () => {
    // Stale cache says admin, DB says user — DB must win.
    getSessionCookie.mockReturnValue("session-token");
    getCookieCache.mockResolvedValue({ user: { role: "admin" } });
    const fetchSpy = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ user: { role: "user" } }), {
        status: 200,
        headers: { "content-type": "application/json" },
      }),
    );

    const response = await authMiddleware(makeRequest("/admin"));

    expect(response.headers.get("location")).toBe("http://localhost:3000/");
    fetchSpy.mockRestore();
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
});
