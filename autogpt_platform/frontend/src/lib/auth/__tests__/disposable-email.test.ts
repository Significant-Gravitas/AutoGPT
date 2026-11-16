// @vitest-environment node
import { isAPIError } from "better-auth/api";
import { afterEach, describe, expect, test, vi } from "vitest";
import {
  assertNotDisposableEmail,
  DISPOSABLE_EMAIL_CODE,
} from "../disposable-email";

type Records = Array<{ exchange: string }>;

function mx(records: Records) {
  return vi.fn(async (_domain: string) => records);
}

function dnsWith(resolveMx: (domain: string) => Promise<Records>) {
  return {
    resolveMx,
    resolve4: async () => ["192.0.2.1"],
    resolve6: async () => [],
  };
}

async function codeOf(promise: Promise<unknown>) {
  try {
    await promise;
    return null;
  } catch (error) {
    return isAPIError(error) ? error.body?.code : "not an APIError";
  }
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("assertNotDisposableEmail", () => {
  test.each(["a@mailinator.com", "a@inbox.mailinator.com", "a@YOPmail.com"])(
    "refuses %s, on the list",
    async (email) => {
      await expect(
        codeOf(assertNotDisposableEmail(email, { dns: dnsWith(mx([])) })),
      ).resolves.toBe(DISPOSABLE_EMAIL_CODE);
    },
  );

  test("refuses an unlisted domain whose mail goes to a temporary-inbox host", async () => {
    const resolveMX = mx([{ exchange: "in.mail.tm." }]);

    await expect(
      codeOf(
        assertNotDisposableEmail("a@rotating-inbox.example", {
          dns: dnsWith(resolveMX),
        }),
      ),
    ).resolves.toBe(DISPOSABLE_EMAIL_CODE);
    expect(resolveMX).toHaveBeenCalledWith("rotating-inbox.example");
  });

  test.each([
    "a@duck.com",
    "a@privaterelay.appleid.com",
    "a@simplelogin.com",
    "a@addy.io",
    "a@mozmail.com",
    "a@proton.me",
    "a@example.com",
  ])("accepts %s", async (email) => {
    await expect(
      codeOf(
        assertNotDisposableEmail(email, {
          dns: dnsWith(mx([{ exchange: "mx.example.net" }])),
        }),
      ),
    ).resolves.toBeNull();
  });

  test("lets the sign-up through when DNS fails", async () => {
    const resolveMX = vi.fn(async () => {
      throw Object.assign(new Error("queryMx ESERVFAIL"), {
        code: "ESERVFAIL",
      });
    });

    await expect(
      codeOf(
        assertNotDisposableEmail("a@slow.example", { dns: dnsWith(resolveMX) }),
      ),
    ).resolves.toBeNull();
  });

  test("lets the sign-up through when DNS is too slow", async () => {
    const resolveMX = vi.fn(
      () =>
        new Promise<Records>((resolve) =>
          setTimeout(() => resolve([{ exchange: "in.mail.tm" }]), 1000),
        ),
    );

    await expect(
      codeOf(
        assertNotDisposableEmail("a@slow.example", {
          dns: dnsWith(resolveMX),
          timeoutMs: 20,
        }),
      ),
    ).resolves.toBeNull();
  });

  test("is off when AUTH_ALLOW_DISPOSABLE_EMAILS is true", async () => {
    vi.stubEnv("AUTH_ALLOW_DISPOSABLE_EMAILS", "true");

    await expect(
      codeOf(
        assertNotDisposableEmail("a@mailinator.com", { dns: dnsWith(mx([])) }),
      ),
    ).resolves.toBeNull();
  });
});
