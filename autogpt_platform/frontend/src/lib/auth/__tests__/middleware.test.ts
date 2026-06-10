import { NextRequest } from "next/server";
import { beforeEach, describe, expect, it, vi } from "vitest";

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
    getSessionCookie.mockReturnValue(null);

    const response = await authMiddleware(
      makeRequest("/copilot", { "sb-proj-auth-token": "legacy" }),
    );

    const location = response.headers.get("location");
    expect(location).toContain("/api/auth/supabase-bridge");
    expect(location).toContain(encodeURIComponent("/copilot"));
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

  it("lets admins through to admin pages via the cookie cache", async () => {
    getSessionCookie.mockReturnValue("session-token");
    getCookieCache.mockResolvedValue({ user: { role: "admin" } });

    const response = await authMiddleware(makeRequest("/admin"));

    expect(response.headers.get("location")).toBeNull();
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
