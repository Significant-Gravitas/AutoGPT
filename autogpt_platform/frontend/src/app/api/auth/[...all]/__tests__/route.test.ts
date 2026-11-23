import { describe, expect, it, vi } from "vitest";

// The route module invokes toNextJsHandler while it is being imported, so the
// mocks must exist before the hoisted import executes.
const {
  getHandlerSentinel,
  postHandlerSentinel,
  toNextJsHandlerMock,
  authInstance,
} = vi.hoisted(() => {
  const getHandlerSentinel = vi.fn();
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
  it("exposes the Better Auth Next.js handlers built from the auth instance", () => {
    expect(toNextJsHandlerMock).toHaveBeenCalledWith(authInstance);
    expect(POST).toBe(postHandlerSentinel);
  });

  // The emailed verify link's own GET is covered in verify-email-confirm-flow.
  it("hands any other GET to Better Auth", async () => {
    const request = new Request("http://localhost:3000/api/auth/get-session");
    getHandlerSentinel.mockResolvedValue(new Response("session"));

    const response = await GET(request);

    expect(getHandlerSentinel).toHaveBeenCalledWith(request);
    expect(await response.text()).toBe("session");
  });
});
