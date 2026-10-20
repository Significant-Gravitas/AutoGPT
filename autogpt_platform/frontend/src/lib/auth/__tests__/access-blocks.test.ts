// @vitest-environment node
import { isAPIError } from "better-auth/api";
import { Pool } from "pg";
import { afterAll, beforeAll, describe, expect, test, vi } from "vitest";

vi.mock("@sentry/nextjs", () => ({ captureException: vi.fn() }));

import {
  assertSignInAllowed,
  assertSignUpAllowed,
  getDomainAndParents,
  SIGN_IN_BLOCKED_CODE,
  SIGN_UP_BLOCKED_CODE,
} from "../access-blocks";

async function codeOf(promise: Promise<unknown>) {
  try {
    await promise;
    return null;
  } catch (error) {
    return isAPIError(error) ? error.body?.code : "not an APIError";
  }
}

describe("getDomainAndParents", () => {
  test("lists the domain and each parent, never the bare TLD", () => {
    expect(getDomainAndParents("a@Mail.Example.co.uk")).toEqual([
      "mail.example.co.uk",
      "example.co.uk",
      "co.uk",
    ]);
  });
});

describe("with a table that can't be read", () => {
  test("sign-up and sign-in carry on", async () => {
    const pool = {
      query: vi.fn().mockRejectedValue(
        Object.assign(new Error("relation does not exist"), {
          code: "42P01",
        }),
      ),
    };

    await expect(
      codeOf(
        assertSignUpAllowed(pool, {
          email: "a@example.com",
          ipAddress: "203.0.113.7",
        }),
      ),
    ).resolves.toBeNull();
    await expect(
      codeOf(assertSignInAllowed(pool, "203.0.113.7")),
    ).resolves.toBeNull();
  });
});

// The SQL is the rule, so it runs against a real database: point
// AUTH_TEST_DATABASE_URL at a throwaway Postgres with the backend migrations
// applied (`?schema=platform` is read from the URL).
const databaseUrl = process.env.AUTH_TEST_DATABASE_URL;

describe.skipIf(!databaseUrl)("against Postgres", () => {
  let pool: Pool;

  beforeAll(async () => {
    const url = new URL(databaseUrl!);
    const schema = url.searchParams.get("schema") ?? "platform";
    url.searchParams.delete("schema");
    pool = new Pool({
      connectionString: url.toString(),
      options: `-c search_path=${schema}`,
    });
    await pool.query(
      `DELETE FROM "UserAuthAccessBlock" WHERE reason = 'vitest'`,
    );
    await pool.query(
      `INSERT INTO "UserAuthAccessBlock" (kind, value, reason, "expiresAt") VALUES
         ('IP', '203.0.113.7', 'vitest', NULL),
         ('IP', '198.51.100.9', 'vitest', now() - interval '1 minute'),
         ('EMAIL_DOMAIN', 'blocked.example', 'vitest', now() + interval '1 day')`,
    );
  });

  afterAll(async () => {
    await pool.query(
      `DELETE FROM "UserAuthAccessBlock" WHERE reason = 'vitest'`,
    );
    await pool.end();
  });

  test.each([
    ["a blocked IP", "a@open.example", "203.0.113.7", SIGN_UP_BLOCKED_CODE],
    [
      "a blocked domain",
      "a@blocked.example",
      "192.0.2.1",
      SIGN_UP_BLOCKED_CODE,
    ],
    [
      "a subdomain of one",
      "a@mail.blocked.example",
      null,
      SIGN_UP_BLOCKED_CODE,
    ],
    ["a lapsed block", "a@open.example", "198.51.100.9", null],
    ["a lookalike domain", "a@notblocked.example", "192.0.2.1", null],
  ])("sign-up from %s", async (_, email, ipAddress, expected) => {
    await expect(
      codeOf(assertSignUpAllowed(pool, { email, ipAddress })),
    ).resolves.toBe(expected);
  });

  test.each([
    ["a blocked IP", "203.0.113.7", SIGN_IN_BLOCKED_CODE],
    ["a lapsed block", "198.51.100.9", null],
    ["an unblocked IP", "192.0.2.1", null],
  ])("sign-in from %s", async (_, ipAddress, expected) => {
    await expect(codeOf(assertSignInAllowed(pool, ipAddress))).resolves.toBe(
      expected,
    );
  });
});
