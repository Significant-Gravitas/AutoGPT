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
    const error = Object.assign(
      new Error(
        'duplicate key value violates unique constraint "User_email_key"',
      ),
      {
        code: "23505",
        constraint: "User_email_key",
        detail: "Key (email)=(dupe@b.c) already exists.",
      },
    );
    const query = vi.fn().mockRejectedValue(error);
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      provisionPlatformUser({ query }, { id: "user-1", email: "dupe@b.c" }),
    ).resolves.toBe("failed");

    expect(errorSpy).toHaveBeenCalledTimes(1);
    expect(captureExceptionMock).toHaveBeenCalledTimes(1);
    const [reported, context] = captureExceptionMock.mock.calls[0];
    // The SQLSTATE and constraint are enough to act on ...
    expect(reported).toBeInstanceOf(Error);
    expect(reported.message).toContain("23505");
    expect(reported.message).toContain("User_email_key");
    expect(context).toEqual(
      expect.objectContaining({
        tags: expect.objectContaining({ pg_code: "23505" }),
        extra: { userId: "user-1", constraint: "User_email_key" },
      }),
    );
    // ... and the raw pg error, whose `detail` carries the email, must reach
    // neither Sentry nor the server log.
    expect(reported).not.toBe(error);
    expect(JSON.stringify([reported.message, context])).not.toContain(
      "dupe@b.c",
    );
    expect(JSON.stringify(errorSpy.mock.calls)).not.toContain("dupe@b.c");
  });

  test("reports an unrecognised failure without inventing pg fields", async () => {
    const query = vi.fn().mockRejectedValue("connection reset");
    vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      provisionPlatformUser({ query }, { id: "user-1", email: "a@b.c" }),
    ).resolves.toBe("failed");

    const [reported, context] = captureExceptionMock.mock.calls[0];
    expect(reported.message).toContain("pg unknown, unknown");
    expect(context.tags.pg_code).toBe("unknown");
  });
});
