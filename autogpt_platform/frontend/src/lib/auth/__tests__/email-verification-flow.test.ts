import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database, so the HTTP behaviour of sign-up, sign-in
// and the verify link is checked end to end rather than assumed.

vi.mock("pg", () => ({ Pool: vi.fn() }));
// ../auth builds the real Better Auth instance, on the in-memory database of
// the test that imported it, so the hooks that reach back into it work too.
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
// Better Auth hands each email it sends to `after` (see background-tasks.ts);
// these are the sends still pending once a response is back.
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

const provisionPlatformUser = vi.hoisted(() => vi.fn());
vi.mock("../provision-platform-user", () => ({ provisionPlatformUser }));

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

  const db = {
    UserAuthIdentity: [] as Array<Record<string, unknown>>,
    UserAuthSession: [] as Array<Record<string, unknown>>,
    UserAuthAccount: [] as Array<Record<string, unknown>>,
    UserAuthVerification: [] as Array<Record<string, unknown>>,
  };
  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler as Handler, db };
}

// A send can queue another (the repeat sign-up's reset email), so drain until
// nothing is left.
async function emailsSent() {
  while (pendingAfterResponse.length) {
    await Promise.all(pendingAfterResponse.splice(0));
  }
}

function never() {
  return new Promise<void>(() => {});
}

async function answeredWithin<T>(response: Promise<T>, ms = 2000) {
  const timedOut = Symbol("timed out");
  const result = await Promise.race([
    response,
    new Promise<typeof timedOut>((resolve) =>
      setTimeout(() => resolve(timedOut), ms),
    ),
  ]);
  if (result === timedOut) throw new Error(`No response within ${ms}ms`);
  return result as T;
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

function signIn(handler: Handler, email: string, as = password) {
  return post(handler, "/sign-in/email", {
    email,
    password: as,
    callbackURL,
  });
}

function signUpAs(handler: Handler, email: string, as: string) {
  return post(handler, "/sign-up/email", {
    email,
    password: as,
    name: email.split("@")[0],
    callbackURL,
  });
}

function lastResetToken(to: string) {
  const email = sentEmails.filter(
    (sent) => sent.to === to && sent.type === "reset_password",
  );
  const url = email.at(-1)?.url;
  return url ? new URL(url).pathname.split("/").at(-1) : undefined;
}

function lastVerifyLink(to: string) {
  const email = sentEmails.filter(
    (sent) => sent.to === to && sent.type === "verify_email",
  );
  return email.at(-1)?.url;
}

beforeEach(() => {
  sentEmails.length = 0;
  pendingAfterResponse.length = 0;
  provisionPlatformUser.mockReset();
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
    await emailsSent();
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
    await emailsSent();
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
    await emailsSent();
    expect(sentEmails).toEqual([]);
  });

  it("verify link signs the user in and lands on /auth/callback", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "new@example.com");
    await emailsSent();

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

  it("creates no platform User row until the link is opened", async () => {
    const { handler } = await createAuthHandler(true);

    await signUp(handler, "new@example.com");
    await emailsSent();
    await handler(new Request(lastVerifyLink("new@example.com") ?? ""));

    // Neither the sign-up nor the verify link provisions: /auth/callback
    // does, through POST /auth/user, which also answers "created" for the
    // sign-up conversion.
    expect(provisionPlatformUser).not.toHaveBeenCalled();
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

  it("emails a reset link, not a verify link, when an unverified address signs up again", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "again@example.com");
    await emailsSent();
    sentEmails.length = 0;

    // Better Auth answers a repeat sign-up like a new one, and the page then
    // says to check the inbox, so something has to arrive.
    const response = await signUp(handler, "again@example.com");

    expect(response.status).toBe(200);
    expect((await response.json()).token).toBeNull();
    expect(db.UserAuthIdentity).toHaveLength(1);
    await emailsSent();
    expect(sentEmails.map((sent) => sent.type)).toEqual(["reset_password"]);
    expect(lastResetToken("again@example.com")).toEqual(expect.any(String));
  });

  it("leaves whoever signed up first no password once the owner verifies", async () => {
    // Someone signs up with an address they do not own...
    const { handler, db } = await createAuthHandler(true);
    await signUpAs(
      handler,
      "victim@example.com",
      "the-first-sign-ups-password",
    );
    await emailsSent();
    const firstLink = lastVerifyLink("victim@example.com") ?? "";
    // ...then the owner signs up too, with their own password.
    await signUpAs(handler, "victim@example.com", "the-owners-own-password");
    await emailsSent();

    // Even if the owner opens the first sign-up's link, which signs them in,
    // the first password must no longer work.
    await handler(new Request(firstLink));
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: true }),
    ]);
    const response = await signIn(
      handler,
      "victim@example.com",
      "the-first-sign-ups-password",
    );
    expect(response.status).toBe(401);
  });

  it("lets the owner set a password through the reset link, which verifies the address", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUpAs(handler, "owner@example.com", "the-first-sign-ups-password");
    await signUp(handler, "owner@example.com");
    await emailsSent();

    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("owner@example.com"),
      newPassword: "the-owners-new-password",
    });

    expect(reset.status).toBe(200);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: true }),
    ]);
    const asOwner = await signIn(
      handler,
      "owner@example.com",
      "the-owners-new-password",
    );
    expect(asOwner.status).toBe(200);
    const asFirst = await signIn(
      handler,
      "owner@example.com",
      "the-first-sign-ups-password",
    );
    expect(asFirst.status).toBe(401);
  });

  it("emails a repeatedly signed-up address once per cooldown", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "spam@example.com");
    await emailsSent();
    sentEmails.length = 0;

    for (let i = 0; i < 3; i++) {
      const response = await signUp(handler, "spam@example.com");
      expect(response.status).toBe(200);
      await emailsSent();
    }
    expect(sentEmails.map((sent) => sent.type)).toEqual(["reset_password"]);

    // Once the window has passed, the next repeat sign-up emails again.
    for (const row of db.UserAuthVerification) {
      if (String(row.identifier).startsWith("repeat-sign-up:")) {
        row.expiresAt = new Date(Date.now() - 1000);
      }
    }
    await signUp(handler, "spam@example.com");
    await emailsSent();
    expect(sentEmails.map((sent) => sent.type)).toEqual([
      "reset_password",
      "reset_password",
    ]);
  });

  it("answers a repeat sign-up like a new one even when its email fails", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "again@example.com");
    await emailsSent();
    const { sendAuthEmail } = await import("../email");
    vi.mocked(sendAuthEmail).mockRejectedValueOnce(new Error("mailer down"));

    const response = await signUp(handler, "again@example.com");
    await emailsSent();

    expect(response.status).toBe(200);
    expect((await response.json()).token).toBeNull();
    expect(db.UserAuthIdentity).toHaveLength(1);
  });

  it("sends nothing when a verified address signs up again", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "taken@example.com");
    await emailsSent();
    await handler(new Request(lastVerifyLink("taken@example.com") ?? ""));
    sentEmails.length = 0;

    const response = await signUp(handler, "taken@example.com");

    expect(response.status).toBe(200);
    expect((await response.json()).token).toBeNull();
    await emailsSent();
    expect(sentEmails).toEqual([]);
  });

  it.each([
    ["a new address", false],
    ["an address still waiting on its link", true],
  ])(
    "answers a sign-up for %s without waiting on the email",
    async (_, alreadySignedUp) => {
      const { handler } = await createAuthHandler(true);
      if (alreadySignedUp) {
        await signUp(handler, "slow@example.com");
        await emailsSent();
      }
      const { sendAuthEmail } = await import("../email");
      vi.mocked(sendAuthEmail).mockImplementationOnce(never);
      const callsBefore = vi.mocked(sendAuthEmail).mock.calls.length;

      // A mail call that never finishes must not hold the response, or its
      // timing tells an unverified address from a verified one.
      const response = await answeredWithin(
        signUp(handler, "slow@example.com"),
      );

      expect(response.status).toBe(200);
      expect((await response.json()).token).toBeNull();
      await vi.waitFor(() =>
        expect(vi.mocked(sendAuthEmail).mock.calls.length).toBeGreaterThan(
          callsBefore,
        ),
      );
      expect(pendingAfterResponse).not.toHaveLength(0);
    },
  );

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

  it("creates the platform User row with the identity, before the session is used", async () => {
    const { handler } = await createAuthHandler(false);

    await signUp(handler, "new@example.com");

    expect(provisionPlatformUser).toHaveBeenCalledTimes(1);
    expect(provisionPlatformUser.mock.calls[0][1]).toEqual(
      expect.objectContaining({
        email: "new@example.com",
        emailVerified: false,
      }),
    );
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
