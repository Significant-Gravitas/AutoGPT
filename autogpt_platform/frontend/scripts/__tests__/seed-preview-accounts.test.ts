import { describe, expect, it } from "vitest";
import {
  PREVIEW_ACCOUNTS,
  assertSafeSchemaName,
  deterministicUserId,
} from "../seed-preview-accounts.helpers";

describe("preview account roster", () => {
  it("seeds the five standard accounts on the previews subdomain", () => {
    expect(PREVIEW_ACCOUNTS).toHaveLength(5);
    for (const account of PREVIEW_ACCOUNTS) {
      expect(account.email.endsWith("@previews.agpt.co")).toBe(true);
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
  it("matches Postgres md5(email)::uuid::text, which older seeded databases carry", () => {
    // Pinned literal: changing the derivation would make re-seeds collide on
    // the email unique constraint in any branch DB seeded before the change.
    expect(deterministicUserId("preview-admin@previews.agpt.co")).toBe(
      "6d08c936-9f91-dadf-0744-a7c3789b322c",
    );
  });

  it("is stable and uuid-shaped for every roster account", () => {
    for (const account of PREVIEW_ACCOUNTS) {
      const id = deterministicUserId(account.email);
      expect(id).toBe(deterministicUserId(account.email));
      expect(id).toMatch(
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
