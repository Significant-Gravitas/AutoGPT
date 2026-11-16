import { APIError } from "better-auth/api";
import { beforeEach, describe, expect, it, vi } from "vitest";

const getSession = vi.hoisted(() => vi.fn());
vi.mock("@/lib/auth/auth", () => ({ auth: { api: { getSession } } }));
vi.mock("next/headers", () => ({ headers: vi.fn(async () => new Headers()) }));

// The suite-wide setup mocks this module; these tests need the real one.
const { getServerSession, isSessionRejection, readServerSession } =
  await vi.importActual<typeof import("../getServerSession")>(
    "../getServerSession",
  );

beforeEach(() => {
  getSession.mockReset();
});

describe("isSessionRejection", () => {
  it("is true only for the auth backend rejecting the session (401)", () => {
    expect(isSessionRejection(new APIError("UNAUTHORIZED"))).toBe(true);
    expect(isSessionRejection(new APIError("INTERNAL_SERVER_ERROR"))).toBe(
      false,
    );
    expect(isSessionRejection(new TypeError("fetch failed"))).toBe(false);
  });
});

describe("reading the session", () => {
  it("lets a backend failure through to callers that must tell it apart", async () => {
    getSession.mockRejectedValue(new APIError("INTERNAL_SERVER_ERROR"));

    await expect(readServerSession()).rejects.toBeInstanceOf(APIError);
  });

  it("still swallows it for callers that only need a session or nothing", async () => {
    vi.spyOn(console, "error").mockImplementation(() => undefined);
    getSession.mockRejectedValue(new Error("db down"));

    await expect(getServerSession()).resolves.toBeNull();
  });
});
