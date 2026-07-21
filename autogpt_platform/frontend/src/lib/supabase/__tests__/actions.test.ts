import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const { cookieJar, mockSignOut } = vi.hoisted(() => ({
  cookieJar: new Map<string, string>(),
  mockSignOut: vi.fn(),
}));

vi.mock("next/headers", () => ({
  cookies: async () => ({
    getAll: () =>
      [...cookieJar.keys()].map((name) => ({
        name,
        value: cookieJar.get(name) ?? "",
      })),
    delete: (name: string) => {
      cookieJar.delete(name);
    },
  }),
}));

vi.mock("../server/getServerSupabase", () => ({
  getServerSupabase: async () => ({ auth: { signOut: mockSignOut } }),
}));

vi.mock("@sentry/nextjs", () => ({
  withServerActionInstrumentation: (
    _name: string,
    _options: unknown,
    fn: () => unknown,
  ) => fn(),
}));

import { serverLogout } from "../actions";

function seedCookies() {
  cookieJar.set("sb-abc-auth-token", "session");
  cookieJar.set("sb-abc-auth-token.0", "chunk-0");
  cookieJar.set("sb-abc-auth-token.1", "chunk-1");
  cookieJar.set("unrelated-cookie", "keep-me");
}

beforeEach(() => {
  cookieJar.clear();
  seedCookies();
});

afterEach(() => {
  vi.clearAllMocks();
});

describe("serverLogout", () => {
  it("clears all sb-* auth cookies on successful sign-out", async () => {
    mockSignOut.mockResolvedValue({ error: null });

    const result = await serverLogout();

    expect(result).toEqual({ success: true });
    expect([...cookieJar.keys()]).toEqual(["unrelated-cookie"]);
  });

  it("clears auth cookies even when signOut returns an error, so the dead session cannot resurrect", async () => {
    mockSignOut.mockResolvedValue({
      error: { message: "session_not_found" },
    });

    const result = await serverLogout();

    expect(result).toEqual({ success: true });
    expect([...cookieJar.keys()]).toEqual(["unrelated-cookie"]);
  });

  it("clears auth cookies even when signOut throws", async () => {
    mockSignOut.mockRejectedValue(new Error("network down"));

    const result = await serverLogout();

    expect(result).toEqual({ success: true });
    expect([...cookieJar.keys()]).toEqual(["unrelated-cookie"]);
  });

  it("passes the global scope through to signOut", async () => {
    mockSignOut.mockResolvedValue({ error: null });

    await serverLogout({ globalLogout: true });

    expect(mockSignOut).toHaveBeenCalledWith({ scope: "global" });
  });
});
