import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIGN_IN_BLOCKED_CODE, SIGN_UP_BLOCKED_CODE } from "../access-blocks";
import { Resolver } from "node:dns/promises";
import { DISPOSABLE_EMAIL_CODE } from "../disposable-email";
import { UNDELIVERABLE_EMAIL_CODE } from "../undeliverable-email";

// Runs the real Better Auth handler with this app's auth options, as
// email-verification-flow.test.ts does, to prove the hooks refuse and log
// where they should. The SQL functions they call are tested against Postgres
// by the backend (data/access_blocks_test.py, data/auth_ip_events_test.py).

const blocks = vi.hoisted(() => ({
  ips: [] as string[],
  domains: [] as string[],
}));
const loggedIps = vi.hoisted(
  () => [] as Array<{ userId: string; kind: string; ip: string }>,
);
vi.mock("pg", () => ({
  Pool: class {
    async query(sql: string, params: unknown[]) {
      if (sql.includes("is_auth_access_blocked(")) {
        const [ip, domains] = params as [string, string[]];
        const blocked =
          blocks.ips.includes(ip) ||
          domains.some((domain) => blocks.domains.includes(domain));
        return { rowCount: 1, rows: [{ blocked }] };
      }
      if (sql.includes("record_auth_ip_event(")) {
        const [userId, kind, ip] = params as string[];
        loggedIps.push({ userId, kind, ip });
        return { rowCount: 1, rows: [] };
      }
      throw new Error(`unexpected query: ${sql}`);
    }
  },
}));

const testDB = vi.hoisted(() => ({ current: {} as Record<string, unknown[]> }));
// Better Auth limits only in production unless told to.
const rateLimit = vi.hoisted(() => ({ enabled: false }));
vi.mock("better-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth")>();
  const { memoryAdapter } = await import("better-auth/adapters/memory");
  return {
    ...actual,
    betterAuth: (options: Parameters<typeof actual.betterAuth>[0]) =>
      actual.betterAuth({
        ...options,
        database: memoryAdapter(testDB.current),
        plugins: [],
        rateLimit: { ...options.rateLimit, enabled: rateLimit.enabled },
      }),
  };
});
vi.mock("next/server", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/server")>()),
  after: vi.fn(),
}));
vi.mock("better-auth/next-js", () => ({
  nextCookies: vi.fn(() => ({ id: "next-cookies" })),
}));
vi.mock("better-auth/plugins", () => ({
  admin: vi.fn(() => ({ id: "admin" })),
  jwt: vi.fn(() => ({ id: "jwt" })),
}));
vi.mock("../supabase-bridge", () => ({
  supabaseBridge: vi.fn(() => ({ id: "supabase-bridge" })),
}));
vi.mock("../email-mirror", () => ({
  mirrorVerifiedEmailToPlatformUser: vi.fn(),
}));
vi.mock("../provision-platform-user", () => ({
  provisionPlatformUser: vi.fn(async () => "created"),
  platformUserExists: vi.fn(async () => true),
}));
vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));

const baseURL = "http://localhost:3000";
const password = "a-long-enough-password"; // pragma: allowlist secret

type Handler = (request: Request) => Promise<Response>;

async function createAuthHandler({ requireEmailVerification = false } = {}) {
  vi.stubEnv(
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    String(requireEmailVerification),
  );
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.doUnmock("../auth");
  vi.resetModules();
  testDB.current = {
    UserAuthIdentity: [],
    UserAuthSession: [],
    UserAuthAccount: [],
    UserAuthVerification: [],
  };
  const { auth } = await import("../auth");
  return auth.handler as Handler;
}

function post(handler: Handler, path: string, body: object, ip: string) {
  return handler(
    new Request(`${baseURL}/api/auth${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: baseURL,
        "x-forwarded-for": ip,
      },
      body: JSON.stringify(body),
    }),
  );
}

function signUp(handler: Handler, email: string, ip: string) {
  return post(
    handler,
    "/sign-up/email",
    { email, password, name: email.split("@")[0] },
    ip,
  );
}

function signIn(handler: Handler, email: string, ip: string) {
  return post(handler, "/sign-in/email", { email, password }, ip);
}

async function refusal(response: Response) {
  return { status: response.status, code: (await response.json()).code };
}

describe("IP and email domain blocks", () => {
  let handler: Handler;

  beforeEach(async () => {
    blocks.ips = [];
    blocks.domains = [];
    loggedIps.length = 0;
    handler = await createAuthHandler();
  });

  afterEach(() => {
    vi.unstubAllEnvs();
  });

  it("refuses a sign-up from a blocked IP and creates no account", async () => {
    blocks.ips = ["203.0.113.7"];

    const response = await signUp(handler, "new@example.com", "203.0.113.7");

    expect(await refusal(response)).toEqual({
      status: 403,
      code: SIGN_UP_BLOCKED_CODE,
    });
    expect(testDB.current.UserAuthIdentity).toHaveLength(0);
  });

  it("refuses a sign-up on a blocked email domain", async () => {
    blocks.domains = ["blocked.example"];

    const response = await signUp(
      handler,
      "new@mail.blocked.example",
      "192.0.2.1",
    );

    expect(await refusal(response)).toEqual({
      status: 403,
      code: SIGN_UP_BLOCKED_CODE,
    });
  });

  it("refuses a sign-up on a disposable email domain", async () => {
    const response = await signUp(handler, "new@mailinator.com", "192.0.2.1");

    expect(await refusal(response)).toEqual({
      status: 403,
      code: DISPOSABLE_EMAIL_CODE,
    });
    expect(testDB.current.UserAuthIdentity).toHaveLength(0);
  });

  it("refuses a sign-up on a domain that takes no mail", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const lookup = vi
      .spyOn(Resolver.prototype, "resolveMx")
      .mockRejectedValue(
        Object.assign(new Error("ENOTFOUND"), { code: "ENOTFOUND" }),
      );

    const response = await signUp(handler, "new@no-mail.example", "192.0.2.1");
    const mxLookups = lookup.mock.calls.length;
    lookup.mockRestore();

    expect(mxLookups).toBe(1);

    expect(response.status).toBe(400);
    expect(await response.json()).toMatchObject({
      code: UNDELIVERABLE_EMAIL_CODE,
      message:
        "We can't deliver email to this address. Check it, or use a different one.",
    });
    expect(testDB.current.UserAuthIdentity).toHaveLength(0);
  });

  it("lets a sign-up through when the mail-server lookup errors", async () => {
    vi.stubEnv("NEXT_PUBLIC_BEHAVE_AS", "CLOUD");
    const lookup = vi
      .spyOn(Resolver.prototype, "resolveMx")
      .mockRejectedValue(
        Object.assign(new Error("ESERVFAIL"), { code: "ESERVFAIL" }),
      );

    const response = await signUp(handler, "new@flaky.example", "192.0.2.1");
    lookup.mockRestore();

    expect(response.status).toBe(200);
  });

  it("answers an admin's domain block before the disposable list", async () => {
    blocks.domains = ["mailinator.com"];

    const response = await signUp(handler, "new@mailinator.com", "192.0.2.1");

    expect(await refusal(response)).toEqual({
      status: 403,
      code: SIGN_UP_BLOCKED_CODE,
    });
  });

  it("refuses a sign-in from a blocked IP for an account made elsewhere", async () => {
    expect((await signUp(handler, "a@example.com", "192.0.2.1")).status).toBe(
      200,
    );
    blocks.ips = ["203.0.113.7"];

    const blocked = await signIn(handler, "a@example.com", "203.0.113.7");
    const elsewhere = await signIn(handler, "a@example.com", "192.0.2.1");

    expect(await refusal(blocked)).toEqual({
      status: 403,
      code: SIGN_IN_BLOCKED_CODE,
    });
    expect(elsewhere.status).toBe(200);
  });

  // With verification required, Better Auth answers a sign-up for an address
  // that has an account without creating one.
  it.each([
    [
      "a blocked domain",
      () => {
        blocks.domains = ["example.com"];
      },
      "a@example.com",
      "b@example.com",
      SIGN_UP_BLOCKED_CODE,
    ],
    [
      "a disposable domain",
      () => vi.stubEnv("AUTH_ALLOW_DISPOSABLE_EMAILS", "false"),
      "a@mailinator.com",
      "b@mailinator.com",
      DISPOSABLE_EMAIL_CODE,
    ],
  ])(
    "refuses a repeat sign-up on %s as it refuses a new one",
    async (_, refuse, existing, fresh, code) => {
      vi.stubEnv("AUTH_ALLOW_DISPOSABLE_EMAILS", "true");
      handler = await createAuthHandler({ requireEmailVerification: true });
      expect((await signUp(handler, existing, "192.0.2.1")).status).toBe(200);
      refuse();

      const repeat = await signUp(handler, existing, "192.0.2.2");
      const first = await signUp(handler, fresh, "192.0.2.3");

      expect(await refusal(repeat)).toEqual({ status: 403, code });
      expect(await refusal(first)).toEqual({ status: 403, code });
    },
  );

  it("logs the sign-up and each sign-in with the client's IP", async () => {
    await signUp(handler, "a@example.com", "192.0.2.1");
    await signIn(handler, "a@example.com", "198.51.100.9");

    const [user] = testDB.current.UserAuthIdentity as Array<{ id: string }>;
    expect(loggedIps).toEqual([
      { userId: user.id, kind: "SIGN_UP", ip: "192.0.2.1" },
      { userId: user.id, kind: "SIGN_IN", ip: "192.0.2.1" },
      { userId: user.id, kind: "SIGN_IN", ip: "198.51.100.9" },
    ]);
  });
});

describe("the client IP", () => {
  beforeEach(() => {
    blocks.ips = ["203.0.113.7"];
    blocks.domains = [];
    loggedIps.length = 0;
    vi.stubEnv("TRUST_X_FORWARDED_FOR", "");
    vi.stubEnv("VERCEL", "");
    vi.stubEnv("NEXT_PUBLIC_VERCEL_ENV", "");
  });

  afterEach(() => {
    rateLimit.enabled = false;
    vi.unstubAllEnvs();
  });

  it("isn't taken from X-Forwarded-For for blocks or the log off Vercel without TRUST_X_FORWARDED_FOR", async () => {
    const handler = await createAuthHandler();

    const signedUp = await signUp(handler, "a@example.com", "203.0.113.7");
    const signedIn = await signIn(handler, "a@example.com", "203.0.113.7");

    expect([signedUp.status, signedIn.status]).toEqual([200, 200]);
    expect(loggedIps).toEqual([]);
  });

  it.each([
    ["on Vercel", "VERCEL", "1"],
    ["with TRUST_X_FORWARDED_FOR", "TRUST_X_FORWARDED_FOR", "true"],
  ])("is taken from X-Forwarded-For for blocks %s", async (_, name, value) => {
    vi.stubEnv(name, value);
    const handler = await createAuthHandler();

    const response = await signUp(handler, "a@example.com", "203.0.113.7");

    expect(await refusal(response)).toEqual({
      status: 403,
      code: SIGN_UP_BLOCKED_CODE,
    });
  });

  it("still rate-limits each address's repeated sign-ins without a trusted proxy", async () => {
    rateLimit.enabled = true;
    const handler = await createAuthHandler();
    await signUp(handler, "a@example.com", "192.0.2.1");

    const statuses = [];
    for (let attempt = 0; attempt < 4; attempt++) {
      statuses.push(
        (await signIn(handler, "a@example.com", "198.51.100.9")).status,
      );
    }
    const elsewhere = await signIn(handler, "a@example.com", "198.51.100.10");

    expect(statuses).toEqual([200, 200, 200, 429]);
    expect(elsewhere.status).toBe(200);
  });
});
