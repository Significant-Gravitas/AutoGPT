// @vitest-environment node
import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  checkEmailDeliverable,
  clearMailServerCache,
  type DnsLookups,
  hasMailServer,
  isDisposableEmailDomain,
} from "../email-deliverability";

function dnsError(code: string) {
  return Object.assign(new Error(code), { code });
}

function failsWith(code: string) {
  return vi.fn(async (_domain: string): Promise<never> => {
    throw dnsError(code);
  });
}

function lookups(overrides: Partial<DnsLookups> = {}) {
  return {
    resolveMx: vi.fn(async (_domain: string) => [
      { exchange: "mx.mail.example" },
    ]),
    resolve4: failsWith("ENODATA"),
    resolve6: failsWith("ENODATA"),
    ...overrides,
  };
}

beforeEach(() => {
  clearMailServerCache();
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("isDisposableEmailDomain", () => {
  test.each(["a@mailinator.com", "a@inbox.mailinator.com", "a@YOPmail.COM"])(
    "matches %s",
    (email) => {
      expect(isDisposableEmailDomain(email)).toBe(true);
    },
  );

  test.each([
    "a@gmail.com",
    "a@notmailinator.example",
    "a@mailinator.com.example",
  ])("does not match %s", (email) => {
    expect(isDisposableEmailDomain(email)).toBe(false);
  });
});

describe("hasMailServer", () => {
  test("is true with an MX record", async () => {
    await expect(hasMailServer("mx.example", { dns: lookups() })).resolves.toBe(
      true,
    );
  });

  test("is false for a null MX", async () => {
    const dns = lookups({ resolveMx: vi.fn(async () => [{ exchange: "" }]) });
    await expect(hasMailServer("null.example", { dns })).resolves.toBe(false);
  });

  test("falls back to an A record when there is no MX", async () => {
    const dns = lookups({
      resolveMx: failsWith("ENODATA"),
      resolve4: vi.fn(async () => ["192.0.2.1"]),
    });
    await expect(hasMailServer("a-only.example", { dns })).resolves.toBe(true);
  });

  test("is false with no MX, A or AAAA record", async () => {
    const dns = lookups({ resolveMx: failsWith("ENODATA") });
    await expect(hasMailServer("empty.example", { dns })).resolves.toBe(false);
  });

  test("is false when the domain does not exist", async () => {
    const dns = lookups({ resolveMx: failsWith("ENOTFOUND") });
    await expect(hasMailServer("nx.example", { dns })).resolves.toBe(false);
  });

  test.each(["ESERVFAIL", "ETIMEOUT", "ECONNREFUSED"])(
    "fails open on %s",
    async (code) => {
      const dns = lookups({ resolveMx: failsWith(code) });
      await expect(hasMailServer("err.example", { dns })).resolves.toBe(true);
    },
  );

  test("fails open when the A lookup errors after no MX", async () => {
    const dns = lookups({
      resolveMx: failsWith("ENODATA"),
      resolve4: failsWith("ESERVFAIL"),
    });
    await expect(hasMailServer("half.example", { dns })).resolves.toBe(true);
  });

  test("fails open when DNS is too slow", async () => {
    const dns = lookups({
      resolveMx: vi.fn(() => new Promise<never>(() => {})),
    });
    await expect(
      hasMailServer("slow.example", { dns, timeoutMs: 10 }),
    ).resolves.toBe(true);
  });

  test("caches a definitive answer, case-insensitively", async () => {
    const dns = lookups({ resolveMx: failsWith("ENOTFOUND") });
    await hasMailServer("cached.example", { dns });
    await expect(hasMailServer("CACHED.example", { dns })).resolves.toBe(false);
    expect(dns.resolveMx).toHaveBeenCalledTimes(1);
  });

  test("does not cache an inconclusive answer", async () => {
    const dns = lookups({ resolveMx: failsWith("ESERVFAIL") });
    await hasMailServer("flaky.example", { dns });
    await hasMailServer("flaky.example", { dns });
    expect(dns.resolveMx).toHaveBeenCalledTimes(2);
  });
});

describe("checkEmailDeliverable", () => {
  test("flags a disposable domain without a lookup", async () => {
    const dns = lookups();
    await expect(
      checkEmailDeliverable("a@mailinator.com", { dns }),
    ).resolves.toEqual({ deliverable: false, reason: "disposable" });
    expect(dns.resolveMx).not.toHaveBeenCalled();
  });

  test("lets a disposable domain through with AUTH_ALLOW_DISPOSABLE_EMAILS", async () => {
    vi.stubEnv("AUTH_ALLOW_DISPOSABLE_EMAILS", "true");
    await expect(
      checkEmailDeliverable("a@mailinator.com", { dns: lookups() }),
    ).resolves.toEqual({ deliverable: true });
  });

  test("flags a domain with no mail server", async () => {
    const dns = lookups({ resolveMx: failsWith("ENOTFOUND") });
    await expect(
      checkEmailDeliverable("a@typo.example", { dns }),
    ).resolves.toEqual({ deliverable: false, reason: "no_mail_server" });
  });

  test("passes a domain with an MX", async () => {
    await expect(
      checkEmailDeliverable("a@real.example", { dns: lookups() }),
    ).resolves.toEqual({ deliverable: true });
  });
});
