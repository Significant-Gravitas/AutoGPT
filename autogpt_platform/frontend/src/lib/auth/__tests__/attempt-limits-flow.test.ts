import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ATTEMPT_LIMITS } from "../attempt-limits";

// Drives the login and reset-password pages' server actions through the real
// Better Auth instance (../auth) on an in-memory database, so the limits are
// checked where the pages reach them (auth.api).

vi.mock("pg", () => ({ Pool: vi.fn() }));
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
const pendingAfterResponse = vi.hoisted(() => [] as Promise<unknown>[]);
vi.mock("next/server", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/server")>()),
  after: vi.fn((task: Promise<unknown>) => {
    pendingAfterResponse.push(task);
  }),
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
const sentEmails = vi.hoisted(
  () => [] as Array<{ to: string; type: string; url: string }>,
);
vi.mock("../email", () => ({
  sendAuthEmail: vi.fn(async (email: (typeof sentEmails)[number]) => {
    sentEmails.push(email);
  }),
}));

// What the pages' server actions read as the incoming request's headers.
const requestHeaders = vi.hoisted(() => ({ current: new Headers() }));
vi.mock("next/headers", () => ({
  headers: vi.fn(async () => requestHeaders.current),
}));
vi.mock("@sentry/nextjs", () => ({
  captureException: vi.fn(),
  withServerActionInstrumentation: (
    _name: string,
    _options: object,
    fn: () => unknown,
  ) => fn(),
}));
vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: vi.fn(async () => ({
    status: 200,
    data: { id: "user-1" },
    headers: new Headers({ "X-AutoGPT-User-Created": "false" }),
  })),
}));
vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: vi.fn(async () => ({ shouldShowOnboarding: false })),
}));
vi.mock("@/lib/auth/server/rollbackSession", () => ({
  rollbackSession: vi.fn(),
}));

const baseURL = "http://localhost:3000";
const password = "a-long-enough-password"; // pragma: allowlist secret
const wrongPassword = "not-the-password"; // pragma: allowlist secret
const INVALID = { success: false, error: "Invalid email or password" };
// The start of a window for every limit, so none rolls over mid-test.
const windowStart = new Date("2026-10-21T12:00:00.000Z");
const signIn = ATTEMPT_LIMITS.get("/sign-in/email")!;
const reset = ATTEMPT_LIMITS.get("/request-password-reset")!;

function refusedFor(wait: string) {
  return `Too many attempts. Please try again in ${wait}.`;
}

async function createAuth({ trustForwardedFor = true } = {}) {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.stubEnv("TRUST_X_FORWARDED_FOR", String(trustForwardedFor));
  vi.doUnmock("../auth");
  vi.resetModules();
  testDB.current = {
    UserAuthIdentity: [],
    UserAuthSession: [],
    UserAuthAccount: [],
    UserAuthVerification: [],
  };
  const { auth } = await import("../auth");
  const { login } = await import("@/app/(no-navbar)/login/actions");
  const { sendResetEmail } = await import(
    "@/app/(platform)/reset-password/actions"
  );
  return {
    api: auth.api,
    handler: auth.handler as (request: Request) => Promise<Response>,
    login,
    sendResetEmail,
  };
}

type API = Awaited<ReturnType<typeof createAuth>>["api"];

function signUp(api: API, email: string) {
  return api.signUpEmail({
    body: { email, password, name: email.split("@")[0] },
  });
}

function fromIP(ip: string) {
  requestHeaders.current = new Headers({ "x-forwarded-for": ip });
}

async function resetEmailsSent() {
  while (pendingAfterResponse.length) {
    await Promise.all(pendingAfterResponse.splice(0));
  }
  return sentEmails.filter((sent) => sent.type === "reset_password");
}

beforeEach(() => {
  sentEmails.length = 0;
  pendingAfterResponse.length = 0;
  requestHeaders.current = new Headers();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(windowStart);
  vi.spyOn(console, "error").mockImplementation(() => undefined);
});

afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllEnvs();
  vi.restoreAllMocks();
});

describe("sign-in through the login page", () => {
  it("refuses guesses at one account from many IPs, then its right password too", async () => {
    const { api, login } = await createAuth();
    await signUp(api, "target@example.com");

    for (let i = 0; i < signIn.perAddress.max; i++) {
      fromIP(`198.51.100.${i + 1}`);
      expect(await login("target@example.com", wrongPassword)).toEqual(INVALID);
    }
    fromIP("198.51.100.100");
    const guess = await login("target@example.com", wrongPassword);
    fromIP("198.51.100.101");
    const owner = await login("target@example.com", password);

    const refused = { success: false, error: refusedFor("10 minutes") };
    expect(guess).toEqual(refused);
    expect(owner).toEqual(refused);
  });

  it("refuses an address with no account exactly like one with an account", async () => {
    const { api, login } = await createAuth();
    await signUp(api, "target@example.com");

    async function refusalFor(email: string) {
      for (let i = 0; i < signIn.perAddress.max; i++) {
        fromIP(`198.51.100.${i + 1}`);
        await login(email, wrongPassword);
      }
      fromIP("198.51.100.100");
      return login(email, wrongPassword);
    }

    expect(await refusalFor("nobody@example.com")).toEqual(
      await refusalFor("target@example.com"),
    );
  });

  it("refuses one IP's guesses across many accounts", async () => {
    const { login } = await createAuth();
    fromIP("198.51.100.7");

    for (let i = 0; i < signIn.perIP.max; i++) {
      expect(await login(`guess-${i}@example.com`, wrongPassword)).toEqual(
        INVALID,
      );
    }

    expect(await login("another@example.com", wrongPassword)).toEqual({
      success: false,
      error: refusedFor("10 seconds"),
    });
  });

  it("leaves other accounts on other IPs signing in", async () => {
    const { api, login } = await createAuth();
    await signUp(api, "target@example.com");
    await signUp(api, "bystander@example.com");
    fromIP("198.51.100.7");
    for (let i = 0; i <= signIn.perAddress.max; i++) {
      await login("target@example.com", wrongPassword);
    }

    fromIP("203.0.113.50");
    expect(await login("bystander@example.com", password)).toEqual({
      success: true,
      next: "/home",
    });
  });

  it("lets the account sign in again once the window has passed", async () => {
    const { api, login } = await createAuth();
    await signUp(api, "target@example.com");
    for (let i = 0; i <= signIn.perAddress.max; i++) {
      fromIP(`198.51.100.${i + 1}`);
      await login("target@example.com", wrongPassword);
    }

    vi.setSystemTime(
      windowStart.getTime() + signIn.perAddress.windowSeconds * 1000,
    );
    fromIP("198.51.100.100");
    expect((await login("target@example.com", password)).success).toBe(true);
  });

  it("limits only the address when the client IP can't be trusted", async () => {
    const { login } = await createAuth({ trustForwardedFor: false });
    fromIP("198.51.100.7");

    const acrossAccounts = [];
    for (let i = 0; i <= signIn.perIP.max; i++) {
      acrossAccounts.push(await login(`guess-${i}@example.com`, wrongPassword));
    }
    for (let i = 0; i < signIn.perAddress.max; i++) {
      await login("target@example.com", wrongPassword);
    }

    expect(acrossAccounts).toEqual(acrossAccounts.map(() => INVALID));
    expect((await login("target@example.com", wrongPassword)).error).toBe(
      refusedFor("10 minutes"),
    );
  });

  it("answers the HTTP route with a 429 and a retry-after", async () => {
    const { api, handler } = await createAuth();
    await signUp(api, "target@example.com");

    const statuses = [];
    let last: Response | undefined;
    for (let i = 0; i <= signIn.perAddress.max; i++) {
      last = await handler(
        new Request(`${baseURL}/api/auth/sign-in/email`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            Origin: baseURL,
            "x-forwarded-for": `198.51.100.${i + 1}`,
          },
          body: JSON.stringify({
            email: "target@example.com",
            password: wrongPassword,
          }),
        }),
      );
      statuses.push(last.status);
    }

    expect(statuses).toEqual([...statuses.slice(0, -1).map(() => 401), 429]);
    expect(last?.headers.get("X-Retry-After")).toBe(String(10 * 60));
  });
});

describe("reset emails through the reset-password page", () => {
  it("sends an address one reset email per window, however many IPs ask", async () => {
    const { api, sendResetEmail } = await createAuth();
    await signUp(api, "inbox@example.com");

    const results = [];
    for (let i = 0; i < 4; i++) {
      fromIP(`203.0.113.${i + 1}`);
      results.push(await sendResetEmail("inbox@example.com"));
    }

    expect(await resetEmailsSent()).toHaveLength(reset.perAddress.max);
    expect(results).toEqual([
      undefined,
      ...Array(3).fill(refusedFor("10 minutes")),
    ]);
  });

  it("refuses an address with no account exactly like one with an account", async () => {
    const { api, sendResetEmail } = await createAuth();
    await signUp(api, "inbox@example.com");

    async function secondAskFor(email: string) {
      fromIP("203.0.113.1");
      await sendResetEmail(email);
      fromIP("203.0.113.2");
      return sendResetEmail(email);
    }

    expect(await secondAskFor("nobody@example.com")).toEqual(
      await secondAskFor("inbox@example.com"),
    );
  });

  it("caps one IP's reset emails across addresses", async () => {
    const { api, sendResetEmail } = await createAuth();
    const addresses = Array.from(
      { length: reset.perIP.max + 2 },
      (_, i) => `user-${i}@example.com`,
    );
    for (const address of addresses) await signUp(api, address);
    fromIP("203.0.113.7");

    const results = [];
    for (const address of addresses) {
      results.push(await sendResetEmail(address));
    }

    expect(await resetEmailsSent()).toHaveLength(reset.perIP.max);
    expect(results.at(-1)).toBe(refusedFor("1 minute"));
  });

  it("leaves other addresses on other IPs alone", async () => {
    const { api, sendResetEmail } = await createAuth();
    await signUp(api, "inbox@example.com");
    await signUp(api, "bystander@example.com");
    fromIP("203.0.113.7");
    for (let i = 0; i <= reset.perIP.max; i++) {
      await sendResetEmail("inbox@example.com");
    }

    fromIP("192.0.2.50");
    expect(await sendResetEmail("bystander@example.com")).toBeUndefined();
    expect((await resetEmailsSent()).map((sent) => sent.to)).toEqual([
      "inbox@example.com",
      "bystander@example.com",
    ]);
  });
});
