import { afterEach, describe, expect, test, vi } from "vitest";
import { mirrorVerifiedEmailToPlatformUser } from "../email-mirror";

describe("mirrorVerifiedEmailToPlatformUser", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  test("writes the verified email to the platform User row, scoped by id and skipping no-ops", async () => {
    const query = vi.fn().mockResolvedValue({ rowCount: 1 });

    await mirrorVerifiedEmailToPlatformUser(
      { query },
      { id: "user-1", email: "new@example.com" },
    );

    expect(query).toHaveBeenCalledTimes(1);
    const [sql, params] = query.mock.calls[0];
    expect(sql).toContain('UPDATE "User"');
    expect(sql).toContain("WHERE id = $2");
    // The `email <> $1` guard keeps the mirror idempotent (no redundant writes).
    expect(sql).toContain("email <> $1");
    expect(params).toEqual(["new@example.com", "user-1"]);
  });

  test("swallows DB errors so a mirror failure never blocks the auth flow", async () => {
    const query = vi.fn().mockRejectedValue(new Error("unique violation"));
    const errorSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    await expect(
      mirrorVerifiedEmailToPlatformUser(
        { query },
        { id: "user-1", email: "dupe@example.com" },
      ),
    ).resolves.toBeUndefined();

    expect(errorSpy).toHaveBeenCalled();
  });
});
