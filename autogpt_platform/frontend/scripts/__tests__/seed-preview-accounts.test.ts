import { createHash } from "node:crypto";
import { describe, expect, it } from "vitest";
import {
  PREVIEW_ACCOUNTS,
  type QueryExecutor,
  assertSafeSchemaName,
  deterministicUserId,
  seedRoster,
} from "../seed-preview-accounts.helpers";

describe("preview account roster", () => {
  it("seeds the five standard accounts on the previews subdomain", () => {
    expect(PREVIEW_ACCOUNTS).toHaveLength(5);
    for (const account of PREVIEW_ACCOUNTS) {
      expect(account.email.endsWith("@previews.agpt.co")).toBe(true);
    }
  });

  it("uses addresses the login page's @agpt.co SSO block does not match", () => {
    // useLoginPage.ts gates on email.includes("@agpt.co"); the roster relies
    // on the subdomain not containing that substring.
    for (const account of PREVIEW_ACCOUNTS) {
      expect(account.email.includes("@agpt.co")).toBe(false);
    }
  });

  it("grants admin only to preview-admin", () => {
    const admins = PREVIEW_ACCOUNTS.filter((a) => a.role === "admin");
    expect(admins.map((a) => a.email)).toEqual([
      "preview-admin@previews.agpt.co",
    ]);
  });
});

describe("deterministicUserId", () => {
  it("matches an independently computed sha256 truncation and the pinned literal", () => {
    const email = "preview-admin@previews.agpt.co";
    const hex = createHash("sha256").update(email).digest("hex").slice(0, 32);
    const independent = [
      hex.slice(0, 8),
      hex.slice(8, 12),
      hex.slice(12, 16),
      hex.slice(16, 20),
      hex.slice(20, 32),
    ].join("-");
    expect(deterministicUserId(email)).toBe(independent);
    // Pinned literal so an accidental derivation change (which would orphan
    // ids the seeder previously inserted) fails loudly.
    expect(deterministicUserId(email)).toBe(
      "5702fe7e-71d4-12ed-0728-436c56f6e8d1",
    );
  });

  it("derives distinct, stable, uuid-shaped ids across the whole roster", () => {
    const ids = PREVIEW_ACCOUNTS.map((a) => deterministicUserId(a.email));
    expect(new Set(ids).size).toBe(PREVIEW_ACCOUNTS.length);
    for (const [i, account] of PREVIEW_ACCOUNTS.entries()) {
      expect(ids[i]).toBe(deterministicUserId(account.email));
      expect(ids[i]).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
      );
    }
  });
});

describe("assertSafeSchemaName", () => {
  it("accepts plain lowercase identifiers", () => {
    expect(assertSafeSchemaName("platform")).toBe("platform");
    expect(assertSafeSchemaName("my_schema2")).toBe("my_schema2");
  });

  it("rejects anything that could break out of an identifier position", () => {
    for (const bad of ['platform"; DROP TABLE x; --', "Platform", "1abc", ""]) {
      expect(() => assertSafeSchemaName(bad)).toThrow(/Unsafe schema name/);
    }
  });
});

const TABLES = {
  identityTable: '"platform"."UserAuthIdentity"',
  accountTable: '"platform"."UserAuthAccount"',
  passwordHash: "$2a$10$fakehashfakehashfakehashfa",
};

interface Call {
  text: string;
  params: unknown[];
}

/**
 * Fake pg client scripted per statement kind. `existingByEmail` maps a
 * roster email to the id the SELECT-by-email returns (simulating rows that
 * pre-exist or were inserted); emails absent from the map behave as freshly
 * inserted (SELECT returns the deterministic id, INSERT reports 1 row).
 */
function fakeClient(behavior: {
  existingByEmail?: Record<string, string | null>;
  existingCredentials?: Set<string>;
}): { client: QueryExecutor; calls: Call[] } {
  const calls: Call[] = [];
  const client: QueryExecutor = {
    async query(text: string, params: unknown[] = []) {
      calls.push({ text, params });
      if (text.includes("INSERT INTO") && text.includes("UserAuthIdentity")) {
        const email = params[2] as string;
        const preexisting = behavior.existingByEmail?.[email] !== undefined;
        return { rows: [], rowCount: preexisting ? 0 : 1 };
      }
      if (text.includes("SELECT id FROM")) {
        const email = params[0] as string;
        if (behavior.existingByEmail?.[email] !== undefined) {
          const id = behavior.existingByEmail[email];
          return { rows: id === null ? [] : [{ id }], rowCount: id ? 1 : 0 };
        }
        return { rows: [{ id: deterministicUserId(email) }], rowCount: 1 };
      }
      if (text.includes("UPDATE")) {
        return { rows: [], rowCount: 0 };
      }
      if (text.includes("INSERT INTO") && text.includes("UserAuthAccount")) {
        const userId = params[0] as string;
        const has = behavior.existingCredentials?.has(userId) ?? false;
        return { rows: [], rowCount: has ? 0 : 1 };
      }
      throw new Error(`Unscripted statement: ${text.slice(0, 60)}`);
    },
  };
  return { client, calls };
}

describe("seedRoster", () => {
  it("creates all five identities and credentials on a fresh database", async () => {
    const { client } = fakeClient({});
    const result = await seedRoster(client, TABLES);
    expect(result).toEqual({ createdIdentities: 5, createdAccounts: 5 });
  });

  it("is idempotent: a second run creates nothing and never rewrites a credential", async () => {
    const existingByEmail = Object.fromEntries(
      PREVIEW_ACCOUNTS.map((a) => [a.email, deterministicUserId(a.email)]),
    );
    const existingCredentials = new Set(Object.values(existingByEmail));
    const { client, calls } = fakeClient({
      existingByEmail,
      existingCredentials,
    });

    const result = await seedRoster(client, TABLES);

    expect(result).toEqual({ createdIdentities: 0, createdAccounts: 0 });
    // The credential statement stays a guarded INSERT — nothing ever issues
    // an UPDATE against the account table, so passwords cannot be rewritten.
    const accountWrites = calls.filter((c) =>
      c.text.includes("UserAuthAccount"),
    );
    for (const write of accountWrites) {
      expect(write.text).toContain("WHERE NOT EXISTS");
      expect(write.text).not.toContain("UPDATE");
    }
  });

  it("attaches the credential to the email-matched id when the identity pre-exists under a different id", async () => {
    const legacyId = "6d08c936-9f91-dadf-0744-a7c3789b322c"; // old md5-derived id
    const { client, calls } = fakeClient({
      existingByEmail: { "preview-admin@previews.agpt.co": legacyId },
    });

    await seedRoster(client, TABLES);

    const credentialInsert = calls.find(
      (c) =>
        c.text.includes("UserAuthAccount") &&
        (c.params[0] as string) === legacyId,
    );
    expect(credentialInsert).toBeDefined();
  });

  it("converges role and emailVerified on the resolved identity", async () => {
    const legacyId = "6d08c936-9f91-dadf-0744-a7c3789b322c";
    const { client, calls } = fakeClient({
      existingByEmail: { "preview-admin@previews.agpt.co": legacyId },
    });

    await seedRoster(client, TABLES);

    const convergence = calls.find(
      (c) => c.text.includes("UPDATE") && (c.params[0] as string) === legacyId,
    );
    expect(convergence).toBeDefined();
    expect(convergence?.params[1]).toBe("admin");
  });

  it("refuses to attach a credential when the deterministic id is taken by a different user", async () => {
    // Identity insert no-ops (id occupied) AND the roster email resolves to
    // nothing — attaching the shared password to the occupying user would be
    // a credential grant to a stranger.
    const { client } = fakeClient({
      existingByEmail: { "preview-admin@previews.agpt.co": null },
    });

    await expect(seedRoster(client, TABLES)).rejects.toThrow(
      /neither existed nor could be created/,
    );
  });
});
