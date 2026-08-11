import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const getServerSessionMock = vi.fn();

vi.mock("../getServerSession", () => ({
  getServerSession: () => getServerSessionMock(),
}));

import { getServerUser } from "../getServerUser";

beforeEach(() => {
  getServerSessionMock.mockReset();
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.restoreAllMocks();
});

describe("getServerUser", () => {
  it("maps the session user and exposes the admin role", async () => {
    getServerSessionMock.mockResolvedValue({
      user: {
        id: "admin-1",
        email: "admin@example.com",
        name: "Admin",
        role: "admin",
        createdAt: new Date("2026-01-02T03:04:05.000Z"),
      },
    });

    const result = await getServerUser();

    expect(result.error).toBeNull();
    expect(result.role).toBe("admin");
    expect(result.user).toEqual({
      id: "admin-1",
      email: "admin@example.com",
      role: "admin",
      created_at: "2026-01-02T03:04:05.000Z",
      user_metadata: { name: "Admin", email: "admin@example.com" },
    });
  });

  it("returns an error when there is no session", async () => {
    getServerSessionMock.mockResolvedValue(null);

    const result = await getServerUser();

    expect(result).toEqual({
      user: null,
      role: null,
      error: "No user found in the response",
    });
  });

  it("catches thrown errors into the error field", async () => {
    getServerSessionMock.mockRejectedValue(new Error("db unavailable"));

    const result = await getServerUser();

    expect(result.user).toBeNull();
    expect(result.role).toBeNull();
    expect(result.error).toBe("Unexpected error: db unavailable");
  });
});
