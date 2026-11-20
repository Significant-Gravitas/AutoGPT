// @vitest-environment node
import { isAPIError } from "better-auth/api";
import { describe, expect, test, vi } from "vitest";

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

describe("with the database's answer", () => {
  function answering(blocked: boolean) {
    return { query: vi.fn().mockResolvedValue({ rows: [{ blocked }] }) };
  }

  test("refuses a blocked sign-up, asking about its IP and every parent domain", async () => {
    const pool = answering(true);

    await expect(
      codeOf(
        assertSignUpAllowed(pool, {
          email: "a@mail.example.com",
          ipAddress: "203.0.113.7",
        }),
      ),
    ).resolves.toBe(SIGN_UP_BLOCKED_CODE);
    expect(pool.query.mock.calls[0][1]).toEqual([
      "203.0.113.7",
      ["mail.example.com", "example.com"],
    ]);
  });

  test("refuses a sign-in from a blocked IP and lets an unblocked one through", async () => {
    await expect(
      codeOf(assertSignInAllowed(answering(true), "203.0.113.7")),
    ).resolves.toBe(SIGN_IN_BLOCKED_CODE);
    await expect(
      codeOf(assertSignInAllowed(answering(false), "203.0.113.7")),
    ).resolves.toBeNull();
  });
});
