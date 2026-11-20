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
    expect(GET).toBe(getHandlerSentinel);
    expect(POST).toBe(postHandlerSentinel);
  });
});
