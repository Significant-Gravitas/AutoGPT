import { describe, expect, it, vi } from "vitest";

// The route module invokes toNextJsHandler while it is being imported, so the
// mocks must exist before the hoisted import executes.
const {
  getHandlerSentinel,
  postHandlerSentinel,
  toNextJsHandlerMock,
  authInstance,
} = vi.hoisted(() => {
  const getHandlerSentinel = vi.fn(async () => Response.json({ token: "jwt" }));
  const postHandlerSentinel = vi.fn();
  return {
    getHandlerSentinel,
    postHandlerSentinel,
    toNextJsHandlerMock: vi.fn(() => ({
      GET: getHandlerSentinel,
      POST: postHandlerSentinel,
    })),
    authInstance: { handler: "better-auth-instance" },
  };
});

vi.mock("better-auth/next-js", () => ({
  toNextJsHandler: toNextJsHandlerMock,
}));

vi.mock("@/lib/auth/auth", () => ({
  auth: authInstance,
}));

import { GET, POST } from "../route";

describe("auth catch-all route", () => {
  it("exposes the Better Auth Next.js handlers built from the auth instance", async () => {
    expect(toNextJsHandlerMock).toHaveBeenCalledWith(authInstance);
    expect(POST).toBe(postHandlerSentinel);

    const request = new Request("http://app/api/auth/get-session");
    await GET(request);
    expect(getHandlerSentinel).toHaveBeenCalledWith(request);
  });

  it("never lets the token response be cached", async () => {
    const response = await GET(new Request("http://app/api/auth/token"));

    expect(response.headers.get("Cache-Control")).toBe("private, no-store");
    expect(await response.json()).toEqual({ token: "jwt" });
  });

  it("leaves other auth responses as Better Auth made them", async () => {
    const response = await GET(new Request("http://app/api/auth/get-session"));

    expect(response.headers.get("Cache-Control")).toBeNull();
  });
});
