// @vitest-environment node
import { APIError } from "better-auth/api";
import { beforeEach, describe, expect, test, vi } from "vitest";

const { getSession, banUser } = vi.hoisted(() => ({
  getSession: vi.fn(),
  banUser: vi.fn(),
}));
vi.mock("@/lib/auth/auth", () => ({
  auth: { api: { getSession, banUser } },
}));

import { maxDuration, POST } from "../route";

function post(body: unknown) {
  return POST(
    new Request("http://localhost/api/admin/ban-accounts", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    }),
  );
}

const ADMIN = { user: { id: "admin-1", role: "admin" } };

describe("POST /api/admin/ban-accounts", () => {
  test("has the function budget of the proxy route, not the platform default", () => {
    expect(maxDuration).toBe(300);
  });

  beforeEach(() => {
    getSession.mockReset();
    banUser.mockReset();
  });

  test.each([
    ["a signed-out caller", null, 401],
    ["a non-admin", { user: { id: "user-1", role: "user" } }, 403],
  ])("refuses %s and bans nobody", async (_, session, status) => {
    getSession.mockResolvedValue(session);

    const response = await post({ userIds: ["u1"], reason: "abuse" });

    expect(response.status).toBe(status);
    expect(banUser).not.toHaveBeenCalled();
  });

  test("bans exactly the listed accounts, with the reason and duration", async () => {
    getSession.mockResolvedValue(ADMIN);
    banUser.mockResolvedValue({ user: {} });

    const response = await post({
      userIds: ["u1", "u2", "u1"],
      reason: "  sign-up farm ",
      banExpiresIn: 3600,
    });

    expect(response.status).toBe(200);
    expect(banUser.mock.calls.map(([args]) => args.body)).toEqual([
      { userId: "u1", banReason: "sign-up farm", banExpiresIn: 3600 },
      { userId: "u2", banReason: "sign-up farm", banExpiresIn: 3600 },
    ]);
    expect((await response.json()).results).toEqual([
      { userId: "u1", ok: true },
      { userId: "u2", ok: true },
    ]);
  });

  test("answers per account when one ban fails", async () => {
    getSession.mockResolvedValue(ADMIN);
    banUser
      .mockResolvedValueOnce({ user: {} })
      .mockRejectedValueOnce(
        new APIError("BAD_REQUEST", { message: "You cannot ban yourself" }),
      )
      .mockResolvedValueOnce({ user: {} });

    const response = await post({
      userIds: ["u1", "admin-1", "u3"],
      reason: "abuse",
    });

    expect((await response.json()).results).toEqual([
      { userId: "u1", ok: true },
      { userId: "admin-1", ok: false, error: "You cannot ban yourself" },
      { userId: "u3", ok: true },
    ]);
  });

  test("refuses a request without a reason", async () => {
    getSession.mockResolvedValue(ADMIN);

    const response = await post({ userIds: ["u1"], reason: "   " });

    expect(response.status).toBe(400);
    expect(banUser).not.toHaveBeenCalled();
  });
});
