import { describe, expect, it, vi } from "vitest";
import { NextRequest } from "next/server";

const updateSessionMock = vi.fn();

vi.mock("@/lib/supabase/middleware", () => ({
  updateSession: (...args: unknown[]) => updateSessionMock(...args),
}));

import { middleware } from "./middleware";

describe("middleware www→non-www redirect", () => {
  it("redirects www host to non-www with 308", async () => {
    const request = new NextRequest("https://www.example.com/dashboard");

    const response = await middleware(request);

    expect(response).toBeDefined();
    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/dashboard",
    );
    expect(updateSessionMock).not.toHaveBeenCalled();
  });

  it("treats uppercase WWW host as case-insensitive (URL API normalizes)", async () => {
    const request = new NextRequest("https://WWW.example.com/path?x=1");

    const response = await middleware(request);

    expect(response.status).toBe(308);
    expect(response.headers.get("location")).toBe(
      "https://example.com/path?x=1",
    );
  });

  it("falls through to updateSession when host is non-www", async () => {
    const passthrough = new Response("ok");
    updateSessionMock.mockResolvedValueOnce(passthrough);

    const request = new NextRequest("https://example.com/dashboard");
    const response = await middleware(request);

    expect(updateSessionMock).toHaveBeenCalledTimes(1);
    expect(response).toBe(passthrough);
  });
});
