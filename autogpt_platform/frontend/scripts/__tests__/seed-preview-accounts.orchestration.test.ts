import { describe, expect, it, vi } from "vitest";
import {
  FAILURE_EXIT_CODE,
  NO_BETTER_AUTH_EXIT_CODE,
  SEEDED_EXIT_CODE,
  reportPreviewSeedFailure,
  resolvePreviewConnectionString,
  runInTransaction,
  seedPreviewAccounts,
  validatePreviewPassword,
} from "../seed-preview-accounts.orchestration";

const OPTIONS = {
  schema: "platform",
  identityTable: '"platform"."UserAuthIdentity"',
  accountTable: '"platform"."UserAuthAccount"',
  password: "twelve-chars", // pragma: allowlist secret
};

function fakePool(tableState: "present" | "absent" | "identity-only") {
  const transactionQueries: string[] = [];
  const release = vi.fn();
  const client = {
    async query(text: string) {
      transactionQueries.push(text);
      return { rows: [], rowCount: 0 };
    },
    release,
  };
  const connect = vi.fn(async () => client);
  const pool = {
    async query(_text: string, params: unknown[] = []) {
      const table = params[0] as string;
      const exists =
        tableState === "present" ||
        (tableState === "identity-only" && table.includes("Identity"));
      return { rows: [{ reg: exists ? table : null }] };
    },
    connect,
  };
  return { pool, client, connect, release, transactionQueries };
}

describe("preview seeder configuration", () => {
  it("pins the cross-repo exit-code contract", () => {
    expect(SEEDED_EXIT_CODE).toBe(0);
    expect(FAILURE_EXIT_CODE).toBe(1);
    expect(NO_BETTER_AUTH_EXIT_CODE).toBe(3);
  });

  it("enforces the 11-reject and 12-accept password boundary", () => {
    expect(() => validatePreviewPassword("x".repeat(11))).toThrow(
      /at least 12 characters/,
    );
    expect(validatePreviewPassword("x".repeat(12))).toBe("x".repeat(12));
  });

  it("rejects missing connection and password configuration", () => {
    expect(() => resolvePreviewConnectionString(undefined, undefined)).toThrow(
      "DIRECT_URL or DATABASE_URL must be set",
    );
    expect(() => validatePreviewPassword(undefined)).toThrow(
      "PREVIEW_ACCOUNTS_PASSWORD must be set",
    );
  });

  it("prefers DIRECT_URL and warns on the DATABASE_URL fallback", () => {
    const reportWarning = vi.fn();
    expect(
      resolvePreviewConnectionString("direct", "pooled", reportWarning),
    ).toBe("direct");
    expect(reportWarning).not.toHaveBeenCalled();
    expect(
      resolvePreviewConnectionString(undefined, "fallback", reportWarning),
    ).toBe("fallback");
    expect(reportWarning).toHaveBeenCalledWith(
      expect.stringContaining("supports an explicit transaction"),
    );
  });

  it("maps a scoped failure message to exit 1", () => {
    const reportError = vi.fn();
    const error = Object.assign(new Error("connection refused"), {
      code: "ECONNREFUSED",
      detail: "secret database detail",
    });

    expect(reportPreviewSeedFailure(error, reportError)).toBe(
      FAILURE_EXIT_CODE,
    );
    expect(reportError).toHaveBeenCalledWith(
      "Preview account seeding failed: connection refused (code ECONNREFUSED)",
    );
  });
});

describe("seedPreviewAccounts", () => {
  it("returns exit 3 when both Better Auth tables are absent", async () => {
    const { pool, connect } = fakePool("absent");
    const reportLog = vi.fn();

    await expect(
      seedPreviewAccounts(pool, OPTIONS, {
        hashPassword: vi.fn(),
        reportLog,
      }),
    ).resolves.toBe(NO_BETTER_AUTH_EXIT_CODE);
    expect(connect).not.toHaveBeenCalled();
    expect(reportLog).toHaveBeenCalledWith(
      expect.stringContaining("pre-migration database"),
    );
  });

  it("fails on a half-applied Better Auth migration", async () => {
    const { pool, connect } = fakePool("identity-only");

    await expect(
      seedPreviewAccounts(pool, OPTIONS, { hashPassword: vi.fn() }),
    ).rejects.toThrow(/half-applied migration/);
    expect(connect).not.toHaveBeenCalled();
  });

  it("returns exit 0 after a committed seed", async () => {
    const { pool, release, transactionQueries } = fakePool("present");
    const seedRoster = vi.fn(async () => ({
      createdIdentities: 5,
      createdAccounts: 5,
    }));
    const reportLog = vi.fn();

    await expect(
      seedPreviewAccounts(pool, OPTIONS, {
        hashPassword: vi.fn(async () => "hash"),
        seedRoster,
        reportLog,
      }),
    ).resolves.toBe(SEEDED_EXIT_CODE);
    expect(transactionQueries).toEqual(["BEGIN", "COMMIT"]);
    expect(release).toHaveBeenCalledOnce();
    expect(reportLog).toHaveBeenCalledWith(
      "Seeded preview accounts: 5 identities and 5 credential accounts created.",
    );
  });

  it("reports an idempotent seed as already up to date", async () => {
    const { pool } = fakePool("present");
    const reportLog = vi.fn();

    await expect(
      seedPreviewAccounts(pool, OPTIONS, {
        hashPassword: vi.fn(async () => "hash"),
        seedRoster: vi.fn(async () => ({
          createdIdentities: 0,
          createdAccounts: 0,
        })),
        reportLog,
      }),
    ).resolves.toBe(SEEDED_EXIT_CODE);
    expect(reportLog).toHaveBeenCalledWith(
      "Preview accounts are now up to date; no new identities or credential accounts created.",
    );
  });
});

describe("runInTransaction", () => {
  it("rolls back and preserves the original operation error", async () => {
    const originalError = new Error("seed failed");
    const queries: string[] = [];
    const client = {
      async query(text: string) {
        queries.push(text);
        return { rows: [], rowCount: 0 };
      },
    };

    await expect(
      runInTransaction(client, async () => {
        throw originalError;
      }),
    ).rejects.toBe(originalError);
    expect(queries).toEqual(["BEGIN", "ROLLBACK"]);
  });

  it("reports rollback failure but still preserves the operation error", async () => {
    const originalError = new Error("seed failed");
    const reportError = vi.fn();
    const client = {
      async query(text: string) {
        if (text === "ROLLBACK") throw new Error("connection lost");
        return { rows: [], rowCount: 0 };
      },
    };

    await expect(
      runInTransaction(
        client,
        async () => {
          throw originalError;
        },
        reportError,
      ),
    ).rejects.toBe(originalError);
    expect(reportError).toHaveBeenCalledWith(
      "Preview account seeder rollback failed: connection lost",
    );
  });
});
