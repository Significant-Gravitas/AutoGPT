import { createHash } from "node:crypto";
import { describe, expect, it, vi } from "vitest";
import {
  PREVIEW_ACCOUNTS,
  type QueryExecutor,
  assertSafeSchemaName,
  closePool,
  deterministicUserID,
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

  it("defines the product state for every preview persona", () => {
    expect(
      PREVIEW_ACCOUNTS.map((account) => ({
        name: account.name,
        subscriptionTier: account.subscriptionTier,
        onboardingComplete: account.onboardingComplete,
      })),
    ).toEqual([
      {
        name: "preview-admin",
        subscriptionTier: "NO_TIER",
        onboardingComplete: true,
      },
      {
        name: "preview-existing",
        subscriptionTier: "NO_TIER",
        onboardingComplete: true,
      },
      {
        name: "preview-clean",
        subscriptionTier: "NO_TIER",
        onboardingComplete: false,
      },
      {
        name: "preview-pro",
        subscriptionTier: "PRO",
        onboardingComplete: true,
      },
      {
        name: "preview-enterprise",
        subscriptionTier: "ENTERPRISE",
        onboardingComplete: true,
      },
    ]);
  });
});

describe("deterministicUserID", () => {
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
    expect(deterministicUserID(email)).toBe(independent);
    // Pinned literal so an accidental derivation change (which would orphan
    // IDs the seeder previously inserted) fails loudly.
    expect(deterministicUserID(email)).toBe(
      "5702fe7e-71d4-12ed-0728-436c56f6e8d1",
    );
  });

  it("derives distinct, stable, uuid-shaped IDs across the whole roster", () => {
    const userIDs = PREVIEW_ACCOUNTS.map((a) => deterministicUserID(a.email));
    expect(new Set(userIDs).size).toBe(PREVIEW_ACCOUNTS.length);
    for (const [i, account] of PREVIEW_ACCOUNTS.entries()) {
      expect(userIDs[i]).toBe(deterministicUserID(account.email));
      expect(userIDs[i]).toMatch(
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

describe("closePool", () => {
  it("preserves the seed result when pool shutdown fails", async () => {
    const reportError = vi.fn();

    await expect(
      closePool(
        {
          async end() {
            throw new Error("shutdown failed");
          },
        },
        reportError,
      ),
    ).resolves.toBeUndefined();
    expect(reportError).toHaveBeenCalledWith(
      "Preview account seeder cleanup failed: shutdown failed",
    );
  });
});

const TABLES = {
  identityTable: '"platform"."UserAuthIdentity"',
  accountTable: '"platform"."UserAuthAccount"',
  userTable: '"platform"."User"',
  profileTable: '"platform"."Profile"',
  onboardingTable: '"platform"."UserOnboarding"',
  subscriptionTierType: '"platform"."SubscriptionTier"',
  onboardingStepType: '"platform"."OnboardingStep"',
  passwordHash: "$2a$10$fakehashfakehashfakehashfa",
};

interface Call {
  text: string;
  params: unknown[];
}

/**
 * Fake pg client mirrors seedRoster's stable operation and table tokens.
 * `existingByEmail` maps a roster email to the id the SELECT-by-email returns
 * (simulating rows that pre-exist or were inserted); emails absent from the
 * map behave as freshly inserted (SELECT returns the deterministic id, INSERT
 * reports 1 row).
 */
function fakeClient(behavior: {
  existingByEmail?: Record<string, string | null>;
  existingCredentials?: Set<string>;
  productUserIDsByEmail?: Record<string, string | null>;
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
      if (
        text.includes("SELECT id FROM") &&
        text.includes("UserAuthIdentity")
      ) {
        const email = params[0] as string;
        if (behavior.existingByEmail?.[email] !== undefined) {
          const id = behavior.existingByEmail[email];
          return { rows: id === null ? [] : [{ id }], rowCount: id ? 1 : 0 };
        }
        return { rows: [{ id: deterministicUserID(email) }], rowCount: 1 };
      }
      if (text.includes("UPDATE") && text.includes("UserAuthIdentity")) {
        return { rows: [], rowCount: 0 };
      }
      if (text.includes("INSERT INTO") && text.includes("UserAuthAccount")) {
        const userID = params[0] as string;
        const has = behavior.existingCredentials?.has(userID) ?? false;
        return { rows: [], rowCount: has ? 0 : 1 };
      }
      if (text.includes("INSERT INTO") && text.includes('"platform"."User"')) {
        const email = params[1] as string;
        const preexisting =
          behavior.productUserIDsByEmail?.[email] !== undefined ||
          behavior.existingByEmail?.[email] !== undefined;
        return { rows: [], rowCount: preexisting ? 0 : 1 };
      }
      if (
        text.includes("SELECT id") &&
        text.includes('"platform"."User"') &&
        !text.includes("UserAuthIdentity")
      ) {
        const email = params[0] as string;
        const configured = behavior.productUserIDsByEmail?.[email];
        if (configured !== undefined) {
          return {
            rows: configured === null ? [] : [{ id: configured }],
            rowCount: configured === null ? 0 : 1,
          };
        }
        const id = behavior.existingByEmail?.[email];
        if (id !== undefined) {
          return { rows: id === null ? [] : [{ id }], rowCount: id ? 1 : 0 };
        }
        return { rows: [{ id: deterministicUserID(email) }], rowCount: 1 };
      }
      if (text.includes("UPDATE") && text.includes('"platform"."User"')) {
        return { rows: [], rowCount: 0 };
      }
      if (
        text.includes("INSERT INTO") &&
        text.includes('"platform"."Profile"')
      ) {
        const preexisting = Object.values(
          behavior.productUserIDsByEmail ?? behavior.existingByEmail ?? {},
        ).includes(params[0] as string);
        return { rows: [], rowCount: preexisting ? 0 : 1 };
      }
      if (text.includes("SELECT id") && text.includes('"platform"."Profile"')) {
        return { rows: [{ id: "profile-id" }], rowCount: 1 };
      }
      if (
        text.includes("INSERT INTO") &&
        text.includes('"platform"."UserOnboarding"')
      ) {
        const preexisting = Object.values(
          behavior.productUserIDsByEmail ?? behavior.existingByEmail ?? {},
        ).includes(params[0] as string);
        return { rows: [], rowCount: preexisting ? 0 : 1 };
      }
      throw new Error(`Unscripted statement: ${text.slice(0, 60)}`);
    },
  };
  return { client, calls };
}

describe("seedRoster", () => {
  it("pre-creates an app user for every persona without writing Stripe customer ids", async () => {
    const { client, calls } = fakeClient({});

    await seedRoster(client, TABLES);

    const userWrites = calls.filter(
      (call) =>
        call.text.includes("INSERT INTO") &&
        call.text.includes('"platform"."User"'),
    );
    expect(userWrites).toHaveLength(PREVIEW_ACCOUNTS.length);
    expect(userWrites.map((write) => write.params[3])).toEqual(
      PREVIEW_ACCOUNTS.map((account) => account.subscriptionTier),
    );
    for (const write of userWrites) {
      expect(write.text).not.toContain("stripeCustomerId");
    }
  });

  it("creates all five identities and credentials on a fresh database", async () => {
    const { client, calls } = fakeClient({});
    const result = await seedRoster(client, TABLES);
    expect(result).toEqual({
      createdIdentities: 5,
      createdAccounts: 5,
      createdUsers: 5,
      updatedUsers: 0,
      createdProfiles: 5,
      changedOnboarding: 5,
    });
    expect(
      calls.filter((call) => call.text.trimStart().startsWith("UPDATE")),
    ).toHaveLength(0);
  });

  it("is idempotent: a second run creates nothing and never rewrites a credential", async () => {
    const existingByEmail = Object.fromEntries(
      PREVIEW_ACCOUNTS.map((a) => [a.email, deterministicUserID(a.email)]),
    );
    const existingCredentials = new Set(Object.values(existingByEmail));
    const { client, calls } = fakeClient({
      existingByEmail,
      existingCredentials,
    });

    const result = await seedRoster(client, TABLES);

    expect(result).toEqual({
      createdIdentities: 0,
      createdAccounts: 0,
      createdUsers: 0,
      updatedUsers: 0,
      createdProfiles: 0,
      changedOnboarding: 0,
    });
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
    const legacyID = "6d08c936-9f91-dadf-0744-a7c3789b322c"; // old md5-derived ID
    const { client, calls } = fakeClient({
      existingByEmail: { "preview-admin@previews.agpt.co": legacyID },
    });

    await seedRoster(client, TABLES);

    const credentialInsert = calls.find(
      (c) =>
        c.text.includes("UserAuthAccount") &&
        (c.params[0] as string) === legacyID,
    );
    expect(credentialInsert).toBeDefined();
  });

  it("converges role and emailVerified on the resolved identity", async () => {
    const legacyID = "6d08c936-9f91-dadf-0744-a7c3789b322c";
    const { client, calls } = fakeClient({
      existingByEmail: { "preview-admin@previews.agpt.co": legacyID },
    });

    await seedRoster(client, TABLES);

    const convergence = calls.find(
      (c) => c.text.includes("UPDATE") && (c.params[0] as string) === legacyID,
    );
    expect(convergence).toBeDefined();
    expect(convergence?.params[1]).toBe("admin");
  });

  it("completes onboarding for established personas but leaves clean incomplete", async () => {
    const { client, calls } = fakeClient({});

    await seedRoster(client, TABLES);

    const onboardingWrites = calls.filter((call) =>
      call.text.includes('"platform"."UserOnboarding"'),
    );
    expect(onboardingWrites).toHaveLength(PREVIEW_ACCOUNTS.length);
    const cleanID = deterministicUserID("preview-clean@previews.agpt.co");
    const cleanWrite = onboardingWrites.find(
      (call) => call.params[0] === cleanID,
    );
    expect(cleanWrite?.text).toContain("array_remove");
    expect(cleanWrite?.text).not.toContain("array_append");
    for (const account of PREVIEW_ACCOUNTS.filter(
      (candidate) => candidate.onboardingComplete,
    )) {
      const userID = deterministicUserID(account.email);
      expect(
        onboardingWrites.find((call) => call.params[0] === userID)?.text,
      ).toContain("array_append");
    }
  });

  it("never writes a Stripe customer id and only converges tiers without one", async () => {
    const existingByEmail = Object.fromEntries(
      PREVIEW_ACCOUNTS.map((account) => [
        account.email,
        deterministicUserID(account.email),
      ]),
    );
    const { client, calls } = fakeClient({ existingByEmail });

    await seedRoster(client, TABLES);

    const productWrites = calls.filter(
      (call) =>
        call.text.includes('"platform"."User"') &&
        !call.text.includes("UserAuthIdentity"),
    );
    for (const write of productWrites) {
      expect(write.text).not.toMatch(/SET\s+"stripeCustomerId"/);
    }
    const tierConvergence = productWrites.find((call) =>
      call.text.includes("SET name = $2"),
    );
    expect(tierConvergence?.text).toContain('WHEN "stripeCustomerId" IS NULL');
    expect(tierConvergence?.text).toContain('ELSE "subscriptionTier"');
  });

  it("refuses an auth and platform user id mismatch for the same email", async () => {
    const email = "preview-admin@previews.agpt.co";
    const { client } = fakeClient({
      existingByEmail: { [email]: deterministicUserID(email) },
      productUserIDsByEmail: {
        [email]: "00000000-0000-0000-0000-000000000001",
      },
    });

    await expect(seedRoster(client, TABLES)).rejects.toThrow(
      /auth identity has id/,
    );
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
