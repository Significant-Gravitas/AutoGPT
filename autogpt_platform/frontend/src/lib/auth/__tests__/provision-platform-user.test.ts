import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

const captureExceptionMock = vi.fn();
vi.mock("@sentry/nextjs", () => ({
  captureException: (...args: unknown[]) => captureExceptionMock(...args),
}));

import { provisionPlatformUser } from "../provision-platform-user";

describe("provisionPlatformUser", () => {
  beforeEach(() => {
    captureExceptionMock.mockReset();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("inserts the platform User row keyed by the auth identity id", async () => {
    const query = vi.fn().mockResolvedValue({ rowCount: 1 });

    await expect(
      provisionPlatformUser(
        { query },
        { id: "user-1", email: "new@example.com", name: "New User" },
      ),
    ).resolves.toBe("created");

    expect(query).toHaveBeenCalledTimes(1);
    const [sql, params] = query.mock.calls[0];
    expect(sql).toContain('INSERT INTO "User"');
    // `updatedAt` is Prisma-managed and has no database default, so the raw
    // insert has to supply it or the statement fails on every sign-up.
    expect(sql).toContain('"updatedAt"');
    // Idempotent against the client's own `POST /auth/user` and the backend's
    // self-heal, both of which may race this hook.
    expect(sql).toContain("ON CONFLICT (id) DO NOTHING");
    expect(params).toEqual(["user-1", "new@example.com", "New User"]);
    expect(captureExceptionMock).not.toHaveBeenCalled();
  });

  test("treats an already-provisioned row as success", async () => {
    const query = vi.fn().mockResolvedValue({ rowCount: 0 });

    await expect(
      provisionPlatformUser({ query }, { id: "user-1", email: "a@b.c" }),
    ).resolves.toBe("exists");

    expect(query.mock.calls[0][1]).toEqual(["user-1", "a@b.c", null]);
    expect(captureExceptionMock).not.toHaveBeenCalled();
  });

  test("never throws: a failed insert is reported and the sign-up continues", async () => {
    // Better Auth awaits create.after hooks post-commit, so a throw here would
    // fail the sign-up after the auth identity is already durable and strand
    // it. Report loudly instead and let the existing provisioning paths run.
    const error = Object.assign(new Error("duplicate key"), { code: "23505" });
    const query = vi.fn().mockRejectedValue(error);
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      provisionPlatformUser({ query }, { id: "user-1", email: "dupe@b.c" }),
    ).resolves.toBe("failed");

    expect(errorSpy).toHaveBeenCalled();
    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    expect(captureExceptionMock).toHaveBeenCalledWith(
      error,
      expect.objectContaining({ extra: { userId: "user-1" } }),
    );
  });
});
