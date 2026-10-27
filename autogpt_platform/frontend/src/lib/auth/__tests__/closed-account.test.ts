import { afterEach, describe, expect, it, vi } from "vitest";

// Runs the real Better Auth handler, the closed-account and admin plugins with
// this app's auth options (from ../auth) on an in-memory database: a deleted
// account is a banned one (backend data/account_deletion.py), and signing in
// to it must say the account is closed, while any other ban says banned.

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
        plugins: (options.plugins ?? []).filter((plugin) =>
          ["closed-account", "admin"].includes(plugin.id),
        ),
      }),
  };
});
vi.mock("better-auth/next-js", () => ({
  nextCookies: vi.fn(() => ({ id: "next-cookies" })),
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

const CLOSED_ACCOUNT_MESSAGE =
  "This account is closed. If you think this is a mistake, contact contact@agpt.co.";
const BANNED_ACCOUNT_MESSAGE =
  "This account has been banned. If you think this is a mistake, please contact support.";
const baseURL = "http://localhost:3000";
const email = "closed@example.com";
const password = "a-long-enough-password";

async function createAuthHandler() {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.doUnmock("../auth");
  vi.resetModules();

  const db = {
    UserAuthIdentity: [] as Array<Record<string, unknown>>,
    UserAuthSession: [] as Array<Record<string, unknown>>,
    UserAuthAccount: [] as Array<Record<string, unknown>>,
    UserAuthVerification: [] as Array<Record<string, unknown>>,
  };
  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler, api: auth.api, db };
}

function signIn(handler: (request: Request) => Promise<Response>) {
  return handler(
    new Request(`${baseURL}/api/auth/sign-in/email`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({ email, password }),
    }),
  );
}

afterEach(() => {
  vi.unstubAllEnvs();
});

async function signUpBanned(banReason: string | null) {
  const auth = await createAuthHandler();
  await auth.api.signUpEmail({ body: { email, password, name: "banned" } });
  for (const user of auth.db.UserAuthIdentity) {
    Object.assign(user, { banned: true, banReason, banExpires: null });
  }
  return auth;
}

describe("signing in to a closed account", () => {
  it("is refused with the closed-account copy", async () => {
    const { handler, db } = await signUpBanned("account_deleted");
    const sessionsBefore = db.UserAuthSession.length;

    const response = await signIn(handler);

    expect(response.status).toBe(403);
    expect(await response.json()).toMatchObject({
      code: "ACCOUNT_CLOSED",
      message: CLOSED_ACCOUNT_MESSAGE,
    });
    expect(db.UserAuthSession).toHaveLength(sessionsBefore);
  });

  it.each([["spam"], [null]])(
    "says banned, not closed, for a ban with reason %s",
    async (banReason) => {
      const { handler, db } = await signUpBanned(banReason);
      const sessionsBefore = db.UserAuthSession.length;

      const response = await signIn(handler);

      expect(response.status).toBe(403);
      expect(await response.json()).toMatchObject({
        code: "BANNED_USER",
        message: BANNED_ACCOUNT_MESSAGE,
      });
      expect(db.UserAuthSession).toHaveLength(sessionsBefore);
    },
  );

  it("still signs an open account in", async () => {
    const { handler, api } = await createAuthHandler();
    await api.signUpEmail({ body: { email, password, name: "open" } });

    const response = await signIn(handler);

    expect(response.status).toBe(200);
  });
});
