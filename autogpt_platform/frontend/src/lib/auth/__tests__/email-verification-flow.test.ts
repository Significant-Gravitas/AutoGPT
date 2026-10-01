import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database, so the HTTP behaviour of sign-up, sign-in
// and the verify link is checked end to end rather than assumed.

vi.mock("pg", () => ({ Pool: vi.fn() }));
// ../auth only builds its options here; the real betterAuth is kept aside so
// each test can build a handler from them on an in-memory database.
vi.mock("better-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth")>();
  return {
    ...actual,
    actualBetterAuth: actual.betterAuth,
    betterAuth: vi.fn((options: unknown) => ({ options })),
  };
});
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

const sentEmails = vi.hoisted(
  () => [] as Array<{ to: string; type: string; url: string }>,
);
vi.mock("../email", () => ({
  sendAuthEmail: vi.fn(async (email: (typeof sentEmails)[number]) => {
    sentEmails.push(email);
  }),
}));

const baseURL = "http://localhost:3000";
const password = "a-long-enough-password";
const callbackURL = "/auth/callback?method=email";

type Handler = (request: Request) => Promise<Response>;

async function createAuthHandler(requireVerification: boolean) {
  vi.stubEnv(
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    requireVerification ? "true" : "false",
  );
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.doUnmock("../auth");
  vi.resetModules();

  const { auth } = (await import("../auth")) as unknown as {
    auth: { options: Record<string, unknown> };
  };
  const { actualBetterAuth } = (await import("better-auth")) as unknown as {
    actualBetterAuth: typeof import("better-auth").betterAuth;
  };
  const { memoryAdapter } = await import("better-auth/adapters/memory");

  const db = {
    UserAuthIdentity: [],
    UserAuthSession: [],
    UserAuthAccount: [],
    UserAuthVerification: [],
  };
  const instance = actualBetterAuth({
    ...auth.options,
    database: memoryAdapter(db),
    plugins: [],
  });
  return { handler: instance.handler as Handler, db };
}

function post(handler: Handler, path: string, body: object) {
  return handler(
    new Request(`${baseURL}/api/auth${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify(body),
    }),
  );
}

function signUp(handler: Handler, email: string) {
  return post(handler, "/sign-up/email", {
    email,
    password,
    name: email.split("@")[0],
    callbackURL,
  });
}

function signIn(handler: Handler, email: string) {
  return post(handler, "/sign-in/email", { email, password, callbackURL });
}

function lastVerifyLink(to: string) {
  const email = sentEmails.filter(
    (sent) => sent.to === to && sent.type === "verify_email",
  );
  return email.at(-1)?.url;
}

beforeEach(() => {
  sentEmails.length = 0;
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("with AUTH_REQUIRE_EMAIL_VERIFICATION=true", () => {
  it("signs up with no session and emails a link back to /auth/callback", async () => {
    const { handler, db } = await createAuthHandler(true);

    const response = await signUp(handler, "new@example.com");

    expect(response.status).toBe(200);
    expect((await response.json()).token).toBeNull();
    expect(db.UserAuthSession).toEqual([]);
    const link = new URL(lastVerifyLink("new@example.com") ?? "");
    expect(link.pathname).toBe("/api/auth/verify-email");
    expect(link.searchParams.get("callbackURL")).toBe(callbackURL);
  });

  it("re-sends the link when an unverified user signs in, instead of a dead end", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "existing@example.com");
    sentEmails.length = 0;

    const response = await signIn(handler, "existing@example.com");

    expect(response.status).toBe(403);
    expect((await response.json()).code).toBe("EMAIL_NOT_VERIFIED");
    expect(lastVerifyLink("existing@example.com")).toContain(
      "/api/auth/verify-email",
    );
  });

  it("does not send a link for a wrong password", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "existing@example.com");
    sentEmails.length = 0;

    const response = await post(handler, "/sign-in/email", {
      email: "existing@example.com",
      password: "not-the-password",
    });

    expect(response.status).toBe(401);
    expect(sentEmails).toEqual([]);
  });

  it("verify link signs the user in and lands on /auth/callback", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "new@example.com");

    const response = await handler(
      new Request(lastVerifyLink("new@example.com") ?? ""),
    );

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(callbackURL);
    // happy-dom strips Set-Cookie from responses, so check the session row.
    expect(db.UserAuthSession).toHaveLength(1);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        email: "new@example.com",
        emailVerified: true,
      }),
    ]);
    // ...and a normal sign-in works from then on.
    expect((await signIn(handler, "new@example.com")).status).toBe(200);
  });

  it("sends an expired or tampered link back to the callback with an error", async () => {
    const { handler } = await createAuthHandler(true);

    const response = await handler(
      new Request(
        `${baseURL}/api/auth/verify-email?token=not-a-token&callbackURL=${encodeURIComponent(callbackURL)}`,
      ),
    );

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(
      `${callbackURL}&error=INVALID_TOKEN`,
    );
  });

  it("re-sends the link through the resend endpoint", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "new@example.com");
    sentEmails.length = 0;

    const response = await post(handler, "/send-verification-email", {
      email: "new@example.com",
      callbackURL,
    });

    expect(response.status).toBe(200);
    expect(lastVerifyLink("new@example.com")).toContain(
      encodeURIComponent(callbackURL),
    );
  });
});

describe("with AUTH_REQUIRE_EMAIL_VERIFICATION off", () => {
  it("signs up straight into a session and sends no email", async () => {
    const { handler, db } = await createAuthHandler(false);

    const response = await signUp(handler, "new@example.com");

    expect(response.status).toBe(200);
    expect((await response.json()).token).toEqual(expect.any(String));
    expect(db.UserAuthSession).toHaveLength(1);
    expect(sentEmails).toEqual([]);
  });
});

describe("team addresses", () => {
  it.each([true, false])(
    "refuses a password sign-up as @agpt.co (verification %s)",
    async (requireVerification) => {
      const { handler, db } = await createAuthHandler(requireVerification);

      const response = await signUp(handler, "made-up@agpt.co");

      expect(response.status).toBe(403);
      expect((await response.json()).code).toBe("TEAM_EMAIL_REQUIRES_GOOGLE");
      expect(db.UserAuthIdentity).toEqual([]);
    },
  );
});
