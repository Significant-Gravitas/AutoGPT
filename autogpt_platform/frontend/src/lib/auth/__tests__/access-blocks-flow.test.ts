import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SIGN_IN_BLOCKED_CODE, SIGN_UP_BLOCKED_CODE } from "../access-blocks";

// Runs the real Better Auth handler with this app's auth options, as
// email-verification-flow.test.ts does, to prove the hooks refuse and log
// where they should. The SQL itself is proven against Postgres in
// access-blocks.test.ts; here the pool answers it from `blocks`.

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
      if (sql.includes('FROM "UserAuthAccessBlock"')) {
        const [ip, domains] = params as [string, string[]];
        const hit =
          blocks.ips.includes(ip) ||
          domains.some((domain) => blocks.domains.includes(domain));
        return { rowCount: hit ? 1 : 0, rows: hit ? [{}] : [] };
      }
      if (sql.includes('INSERT INTO "UserAuthIpEvent"')) {
        const [userId, kind, ip] = params as string[];
        loggedIps.push({ userId, kind, ip });
        return { rowCount: 1, rows: [] };
      }
      throw new Error(`unexpected query: ${sql}`);
    }
  },
}));

const testDB = vi.hoisted(() => ({ current: {} as Record<string, unknown[]> }));
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

async function createAuthHandler() {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
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
