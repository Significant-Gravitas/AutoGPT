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
  it("derives the pinned sha256-based id for a known email", () => {
    // Pinned literal: the id must be stable across runs and machines so a
    // re-run of the seeder computes the same id it inserted last time.
    // (Older md5-seeded databases are handled by the email match, not this.)
    expect(deterministicUserId("preview-admin@previews.agpt.co")).toBe(
      "5702fe7e-71d4-12ed-0728-436c56f6e8d1",
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
