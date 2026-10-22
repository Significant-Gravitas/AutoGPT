import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { AUTH_EMAILS_PER_IP } from "../auth-email-cooldown";

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

// The platform `User` rows the auth hook has written, as the real
// provisionPlatformUser and platformUserExists would see them.
const platformUsers = vi.hoisted(() => new Set<string>());
const userLookupFails = vi.hoisted(() => ({ current: false }));
const provisionPlatformUser = vi.hoisted(() =>
  vi.fn(async (_pool: unknown, user: { id: string }) => {
    platformUsers.add(user.id);
    return "created";
  }),
);
vi.mock("../provision-platform-user", () => ({
  provisionPlatformUser,
  platformUserExists: async (_pool: unknown, userId: string) => {
    if (userLookupFails.current) throw new Error("database is down");
    return platformUsers.has(userId);
  },
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

function emptyDB() {
  return {
    UserAuthIdentity: [] as Array<Record<string, unknown>>,
    UserAuthSession: [] as Array<Record<string, unknown>>,
    UserAuthAccount: [] as Array<Record<string, unknown>>,
    UserAuthVerification: [] as Array<Record<string, unknown>>,
  };
}

// Pass the db of an earlier handler to flip the flag on the same accounts.
async function createAuthHandler(requireVerification: boolean, db = emptyDB()) {
  vi.stubEnv(
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    requireVerification ? "true" : "false",
  );
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.doUnmock("../auth");
  vi.resetModules();

  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler as Handler, api: auth.api, db };
}

type TestDB = Awaited<ReturnType<typeof createAuthHandler>>["db"];

// Lets the per-address email cooldowns lapse, as ten minutes would.
function expireCooldowns(db: TestDB) {
  for (const row of db.UserAuthVerification) {
    if (/^(repeat-sign-up|verify-email):/.test(String(row.identifier))) {
      row.expiresAt = new Date(Date.now() - 1000);
    }
  }
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

type API = Awaited<ReturnType<typeof createAuthHandler>>["api"];

// A real sign-out, which deletes the session row.
async function signUpAndSignOut(api: API, email: string, as: string) {
  const { headers } = await api.signUpEmail({
    body: { email, password: as, name: email.split("@")[0], callbackURL },
    returnHeaders: true,
  });
  const cookie = headers.get("set-cookie")?.split(";")[0] ?? "";
  await api.signOut({ headers: new Headers({ cookie }) });
}

function fromIP(ip: string) {
  return new Headers({ "x-forwarded-for": ip });
}

function lastResetToken(to: string, type = "set_password") {
  const email = sentEmails.filter(
    (sent) => sent.to === to && sent.type === type,
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
  provisionPlatformUser.mockClear();
  platformUsers.clear();
  userLookupFails.current = false;
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
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "existing@example.com");
    await emailsSent();
    sentEmails.length = 0;
    // The sign-up's own email holds the address's cooldown for a while.
    expireCooldowns(db);

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

  it("emails a set-password link, not a verify link, when an unverified address signs up again", async () => {
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
    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);
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

  it("still resets the password when marking the address verified fails", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "unverified@example.com");
    await post(handler, "/request-password-reset", {
      email: "unverified@example.com",
      redirectTo: `${baseURL}/reset-password`,
    });
    await emailsSent();
    const { auth } = await import("../auth");
    const { internalAdapter } = await auth.$context;
    const updateUser = vi
      .spyOn(internalAdapter, "updateUser")
      .mockRejectedValueOnce(new Error("database is down"));
    const consoleError = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});

    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("unverified@example.com", "reset_password"),
      newPassword: "a-new-long-enough-password",
    });

    expect(updateUser).toHaveBeenCalled();
    expect(reset.status).toBe(200);
    expect(JSON.stringify(consoleError.mock.calls)).not.toContain(
      "unverified@example.com",
    );
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: false }),
    ]);
    // The address stays unverified, so signing in with the new password
    // sends a fresh verify link rather than a dead end.
    sentEmails.length = 0;
    expireCooldowns(db);
    const signedIn = await signIn(
      handler,
      "unverified@example.com",
      "a-new-long-enough-password",
    );
    expect(signedIn.status).toBe(403);
    expect((await signedIn.json()).code).toBe("EMAIL_NOT_VERIFIED");
    await emailsSent();
    expect(lastVerifyLink("unverified@example.com")).toContain(
      "/api/auth/verify-email",
    );
    consoleError.mockRestore();
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
    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);

    // Once the window has passed, the next repeat sign-up emails again.
    expireCooldowns(db);
    await signUp(handler, "spam@example.com");
    await emailsSent();
    expect(sentEmails.map((sent) => sent.type)).toEqual([
      "set_password",
      "set_password",
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
    expect(sendAuthEmail).toHaveBeenLastCalledWith(
      expect.objectContaining({
        to: "again@example.com",
        type: "set_password",
      }),
    );
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

  it("keeps the password of an account that has signed in before", async () => {
    // An unverified account from before the flag was turned on: its owner set
    // the password and has used it, so a stranger signing up with the address
    // must not lock them out.
    const { handler, db } = await createAuthHandler(true);
    await signUpAs(handler, "legacy@example.com", "the-owners-own-password");
    await emailsSent();
    const [identity] = db.UserAuthIdentity;
    db.UserAuthSession.push({
      id: "legacy-session",
      userId: identity.id,
      token: "legacy-session-token",
      expiresAt: new Date(Date.now() + 60_000),
      createdAt: new Date(),
      updatedAt: new Date(),
    });
    sentEmails.length = 0;

    await signUpAs(handler, "legacy@example.com", "a-strangers-password");
    await emailsSent();

    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);
    expireCooldowns(db);
    const asOwner = await signIn(
      handler,
      "legacy@example.com",
      "the-owners-own-password",
    );
    expect(asOwner.status).toBe(403);
    expect((await asOwner.json()).code).toBe("EMAIL_NOT_VERIFIED");
  });

  it("keeps the password of an owner from before the flag who has signed out", async () => {
    const before = await createAuthHandler(false);
    await signUpAndSignOut(
      before.api,
      "legacy@example.com",
      "the-owners-own-password",
    );
    expect(before.db.UserAuthSession).toHaveLength(0);
    const { handler, db } = await createAuthHandler(true, before.db);

    await signUpAs(handler, "legacy@example.com", "a-strangers-password");
    await emailsSent();

    expireCooldowns(db);
    const asOwner = await signIn(
      handler,
      "legacy@example.com",
      "the-owners-own-password",
    );
    expect(asOwner.status).toBe(403);
  });

  it("answers a used account's resend with a set-password link, not one that signs in", async () => {
    // Someone registers the owner's address while the flag is off, and so
    // gets a session; then the flag goes on and the owner signs up too.
    const before = await createAuthHandler(false);
    await signUpAs(
      before.handler,
      "owner@example.com",
      "the-first-sign-ups-password",
    );
    const { handler } = await createAuthHandler(true, before.db);
    await signUpAs(handler, "owner@example.com", "the-owners-own-password");
    await emailsSent();
    sentEmails.length = 0;

    const resend = await post(handler, "/send-verification-email", {
      email: "owner@example.com",
      callbackURL,
    });
    await emailsSent();

    expect(resend.status).toBe(200);
    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);
    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("owner@example.com"),
      newPassword: "the-owners-new-password",
    });
    expect(reset.status).toBe(200);
    const asFirst = await signIn(
      handler,
      "owner@example.com",
      "the-first-sign-ups-password",
    );
    expect(asFirst.status).toBe(401);
    const asOwner = await signIn(
      handler,
      "owner@example.com",
      "the-owners-new-password",
    );
    expect(asOwner.status).toBe(200);
  });

  it("leaves the first password dead when the owner signs up, then verifies through Resend", async () => {
    // The owner can only reach Resend through their own sign-up: logging in
    // with their password fails on an account someone else created.
    const { handler } = await createAuthHandler(true);
    await signUpAs(handler, "owner@example.com", "the-first-sign-ups-password");
    await emailsSent();
    const ownersLogin = await signIn(
      handler,
      "owner@example.com",
      "the-owners-own-password",
    );
    expect(ownersLogin.status).toBe(401);
    await signUpAs(handler, "owner@example.com", "the-owners-own-password");
    await emailsSent();

    await post(handler, "/send-verification-email", {
      email: "owner@example.com",
      callbackURL,
    });
    const verified = await handler(
      new Request(lastVerifyLink("owner@example.com") ?? ""),
    );

    expect(verified.status).toBe(302);
    const asFirst = await signIn(
      handler,
      "owner@example.com",
      "the-first-sign-ups-password",
    );
    expect(asFirst.status).toBe(401);
  });

  it("still replaces the first password when the User row can't be read", async () => {
    const { handler } = await createAuthHandler(true);
    await signUpAs(
      handler,
      "victim@example.com",
      "the-first-sign-ups-password",
    );
    await emailsSent();
    const firstLink = lastVerifyLink("victim@example.com") ?? "";
    userLookupFails.current = true;

    await signUpAs(handler, "victim@example.com", "the-owners-own-password");
    await emailsSent();
    await handler(new Request(firstLink));

    expect(sentEmails.at(-1)?.type).toBe("set_password");
    const asFirst = await signIn(
      handler,
      "victim@example.com",
      "the-first-sign-ups-password",
    );
    expect(asFirst.status).toBe(401);
  });

  it("answers Resend with the set-password link when the User row can't be read", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "unknown@example.com");
    await emailsSent();
    sentEmails.length = 0;
    userLookupFails.current = true;

    const resend = await post(handler, "/send-verification-email", {
      email: "unknown@example.com",
      callbackURL,
    });

    expect(resend.status).toBe(200);
    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);
  });

  it("answers Resend with the set-password link when the session lookup fails", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "unknown@example.com");
    await emailsSent();
    sentEmails.length = 0;
    const { auth } = await import("../auth");
    const { adapter } = await auth.$context;
    const findMany = adapter.findMany.bind(adapter);
    let failed = false;
    vi.spyOn(adapter, "findMany").mockImplementation(async (args) => {
      if (args.model === "session" && !failed) {
        failed = true;
        throw new Error("database is down");
      }
      return findMany(args);
    });

    const resend = await post(handler, "/send-verification-email", {
      email: "unknown@example.com",
      callbackURL,
    });

    expect(failed).toBe(true);
    expect(resend.status).toBe(200);
    expect(sentEmails.map((sent) => sent.type)).toEqual(["set_password"]);
  });

  it("lets one IP have only a few addresses emailed through the sign-up page", async () => {
    const { api } = await createAuthHandler(true);

    // The page's server action calls auth.api, which Better Auth's own per-IP
    // limiter never sees. +aliases all reach one mailbox.
    const outcomes = [];
    for (let i = 0; i < 20; i++) {
      outcomes.push(
        await api
          .signUpEmail({
            body: {
              email: `mailbox+${i}@example.com`,
              password,
              name: "mailbox",
              callbackURL,
            },
            headers: fromIP("203.0.113.7"),
          })
          .then(
            () => "sent",
            (error: { statusCode?: number }) => error.statusCode,
          ),
      );
    }
    await emailsSent();

    expect(sentEmails).toHaveLength(AUTH_EMAILS_PER_IP);
    expect(outcomes.filter((outcome) => outcome === 429)).toHaveLength(
      20 - AUTH_EMAILS_PER_IP,
    );
    await api.signUpEmail({
      body: { email: "someone@example.com", password, name: "x", callbackURL },
      headers: fromIP("198.51.100.9"),
    });
    await emailsSent();
    expect(lastVerifyLink("someone@example.com")).toBeDefined();
  });

  it("counts the resend button against the same per-IP cap", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "resends@example.com");

    const statuses = [];
    for (let i = 0; i < AUTH_EMAILS_PER_IP; i++) {
      const response = await post(handler, "/send-verification-email", {
        email: "resends@example.com",
        callbackURL,
      });
      statuses.push(response.status);
    }

    expect(statuses.filter((status) => status === 200)).toHaveLength(
      AUTH_EMAILS_PER_IP - 1,
    );
    expect(statuses.at(-1)).toBe(429);
  });

  it("emails an unverified address once per cooldown however often it signs in", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "signs-in@example.com");
    await emailsSent();
    expireCooldowns(db);
    sentEmails.length = 0;

    for (let i = 0; i < 3; i++) {
      expect((await signIn(handler, "signs-in@example.com")).status).toBe(403);
      await emailsSent();
    }
    expect(sentEmails.map((sent) => sent.type)).toEqual(["verify_email"]);

    expireCooldowns(db);
    await signIn(handler, "signs-in@example.com");
    await emailsSent();
    expect(sentEmails.map((sent) => sent.type)).toEqual([
      "verify_email",
      "verify_email",
    ]);
  });

  it("does not hold back the resend button's emails", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "resend@example.com");
    await emailsSent();
    sentEmails.length = 0;

    for (let i = 0; i < 2; i++) {
      const response = await post(handler, "/send-verification-email", {
        email: "resend@example.com",
        callbackURL,
      });
      expect(response.status).toBe(200);
    }

    expect(sentEmails.map((sent) => sent.type)).toEqual([
      "verify_email",
      "verify_email",
    ]);
  });

  it("leaves a verified account verified after a password reset", async () => {
    const { handler, db } = await createAuthHandler(true);
    await signUp(handler, "verified@example.com");
    await emailsSent();
    await handler(new Request(lastVerifyLink("verified@example.com") ?? ""));
    await post(handler, "/request-password-reset", {
      email: "verified@example.com",
      redirectTo: `${baseURL}/reset-password`,
    });
    await emailsSent();
    const [identity] = db.UserAuthIdentity;
    const untouchedSince = new Date("2020-01-01T00:00:00Z");
    identity.updatedAt = untouchedSince;

    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("verified@example.com", "reset_password"),
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(200);
    // The reset writes the password, not the identity: a verified address
    // has nothing to update.
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        emailVerified: true,
        updatedAt: untouchedSince,
      }),
    ]);
    const signedIn = await signIn(
      handler,
      "verified@example.com",
      "a-new-long-enough-password",
    );
    expect(signedIn.status).toBe(200);
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

  it("does not cap sign-ups per IP", async () => {
    const { handler } = await createAuthHandler(false);

    for (let i = 0; i <= AUTH_EMAILS_PER_IP; i++) {
      expect((await signUp(handler, `user${i}@example.com`)).status).toBe(200);
    }
  });

  it("verifies the address from a resent link without signing its opener in", async () => {
    const { handler, db } = await createAuthHandler(false);
    await signUp(handler, "new@example.com");
    db.UserAuthSession.length = 0;
    await post(handler, "/send-verification-email", {
      email: "new@example.com",
      callbackURL,
    });

    const response = await handler(
      new Request(lastVerifyLink("new@example.com") ?? ""),
    );

    expect(response.status).toBe(302);
    expect(db.UserAuthSession).toHaveLength(0);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: true }),
    ]);
  });

  it("does not hold back a change of email's verification email", async () => {
    // The per-address cap is for sign-up and sign-in, which send nothing with
    // verification off; a retried change of email still gets its link.
    const { api } = await createAuthHandler(false);
    const { headers } = await api.signUpEmail({
      body: { email: "a@example.com", password, name: "a", callbackURL },
      returnHeaders: true,
    });
    const session = new Headers({
      cookie: headers.get("set-cookie")?.split(";")[0] ?? "",
    });

    for (const newEmail of [
      "b@example.com",
      "a@example.com",
      "b@example.com",
    ]) {
      await api.changeEmail({ body: { newEmail }, headers: session });
    }
    await emailsSent();

    const toB = sentEmails.filter(
      (sent) => sent.to === "b@example.com" && sent.type === "verify_email",
    );
    expect(toB).toHaveLength(2);
  });

  it("does not verify an address a reset link was never mailed to", async () => {
    // A reset token names the account, not the address it went to, and an
    // unverified account changes address at once. A verified address would
    // let Better Auth link that address's OAuth sign-in into this account.
    const { api, handler, db } = await createAuthHandler(false);
    const { headers } = await api.signUpEmail({
      body: { email: "attacker@example.com", password, name: "a", callbackURL },
      returnHeaders: true,
    });
    const cookie = headers.get("set-cookie")?.split(";")[0] ?? "";
    await api.requestPasswordReset({
      body: { email: "attacker@example.com", redirectTo: "/reset-password" },
    });
    await emailsSent();
    const token = lastResetToken("attacker@example.com", "reset_password");
    await api.changeEmail({
      body: { newEmail: "victim@example.com" },
      headers: new Headers({ cookie }),
    });

    const response = await post(handler, "/reset-password", {
      newPassword: "a-new-long-enough-password",
      token,
    });

    expect(response.status).toBe(400);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        email: "victim@example.com",
        emailVerified: false,
      }),
    ]);
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

// Signed up with verification off, so the account has a session.
async function signUpSignedIn(api: API, email: string) {
  const { headers } = await api.signUpEmail({
    body: { email, password, name: email.split("@")[0], callbackURL },
    returnHeaders: true,
  });
  return new Headers({
    cookie: headers.get("set-cookie")?.split(";")[0] ?? "",
  });
}

function requestReset(handler: Handler, email: string) {
  return post(handler, "/request-password-reset", {
    email,
    redirectTo: `${baseURL}/reset-password`,
  });
}

// What Better Auth decides when someone signs in with Google as `email`.
async function linkGoogle(email: string) {
  const { auth } = await import("../auth");
  const { handleOAuthUserInfo } = await import("better-auth/oauth2");
  const context = await auth.$context;
  return handleOAuthUserInfo(
    { context } as never,
    {
      userInfo: {
        id: "google-account-id",
        email,
        emailVerified: true,
        name: "Google user",
      },
      account: { providerId: "google", accountId: "google-account-id" },
      callbackURL: "/",
    } as never,
  );
}

describe("password reset links", () => {
  // kcze's attack: the reset link goes to an address the attacker holds,
  // then the account takes the victim's address before the link is opened.
  it.each([false, true])(
    "do not verify an address the link was not mailed to (verification %s)",
    async (requireVerification) => {
      const { api: signUpAPI, db } = await createAuthHandler(false);
      const session = await signUpSignedIn(signUpAPI, "attacker@example.com");
      const { handler, api } = await createAuthHandler(requireVerification, db);
      await requestReset(handler, "attacker@example.com");
      await emailsSent();
      const token = lastResetToken("attacker@example.com", "reset_password");

      // Applied on the spot: the account is unverified.
      await api.changeEmail({
        body: { newEmail: "victim@gmail.com" },
        headers: session,
      });
      await post(handler, "/reset-password", {
        token,
        newPassword: "the-attackers-new-password",
      });

      expect(db.UserAuthIdentity).toEqual([
        expect.objectContaining({
          email: "victim@gmail.com",
          emailVerified: false,
        }),
      ]);
      expect(await linkGoogle("victim@gmail.com")).toEqual(
        expect.objectContaining({ error: "account not linked" }),
      );
    },
  );

  it("stop working once the account's address changes", async () => {
    const { handler, api, db } = await createAuthHandler(false);
    const session = await signUpSignedIn(api, "first@example.com");
    await requestReset(handler, "first@example.com");
    await emailsSent();
    const token = lastResetToken("first@example.com", "reset_password");

    await api.changeEmail({
      body: { newEmail: "second@example.com" },
      headers: session,
    });
    const reset = await post(handler, "/reset-password", {
      token,
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(400);
    expect(
      db.UserAuthVerification.filter((row) =>
        String(row.identifier).startsWith("reset-password"),
      ),
    ).toEqual([]);
  });

  it("stop working once a verified account's change of address is confirmed", async () => {
    const { api: signUpAPI, db } = await createAuthHandler(false);
    const session = await signUpSignedIn(signUpAPI, "old@example.com");
    db.UserAuthIdentity[0].emailVerified = true;
    const { handler, api } = await createAuthHandler(false, db);
    await requestReset(handler, "old@example.com");
    await emailsSent();
    const token = lastResetToken("old@example.com", "reset_password");

    // The address changes on /verify-email, when the new address's link is
    // opened, not on /change-email.
    await api.changeEmail({
      body: { newEmail: "new@example.com", callbackURL: "/" },
      headers: session,
    });
    await emailsSent();
    const confirm = sentEmails.find((sent) => sent.type === "change_email");
    await handler(new Request(confirm?.url ?? "", { headers: session }));
    await emailsSent();
    await handler(
      new Request(lastVerifyLink("new@example.com") ?? "", {
        headers: session,
      }),
    );
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ email: "new@example.com" }),
    ]);

    const reset = await post(handler, "/reset-password", {
      token,
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(400);
    expect(
      db.UserAuthVerification.filter((row) =>
        String(row.identifier).startsWith("reset-password"),
      ),
    ).toEqual([]);
  });

  it("verify only the address they were mailed to, even if the link survives a change", async () => {
    const { handler, db } = await createAuthHandler(false);
    await signUp(handler, "first@example.com");
    await requestReset(handler, "first@example.com");
    await emailsSent();
    // An address change that skipped Better Auth's hooks, e.g. by hand.
    db.UserAuthIdentity[0].email = "second@example.com";

    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("first@example.com", "reset_password"),
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(200);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: false }),
    ]);
  });

  it.each([false, true])(
    "verify the address they were mailed to (verification %s)",
    async (requireVerification) => {
      const { handler: signUpHandler, db } = await createAuthHandler(false);
      await signUp(signUpHandler, "owner@example.com");
      const { handler } = await createAuthHandler(requireVerification, db);
      await requestReset(handler, "owner@example.com");
      await emailsSent();

      const reset = await post(handler, "/reset-password", {
        token: lastResetToken("owner@example.com", "reset_password"),
        newPassword: "a-new-long-enough-password",
      });

      expect(reset.status).toBe(200);
      expect(db.UserAuthIdentity).toEqual([
        expect.objectContaining({ emailVerified: true }),
      ]);
      expect(
        db.UserAuthVerification.filter((row) =>
          String(row.identifier).startsWith("reset-password"),
        ),
      ).toEqual([]);
    },
  );

  it("keep a link whose address wasn't recorded, which then verifies nothing", async () => {
    const { handler, api, db } = await createAuthHandler(false);
    const session = await signUpSignedIn(api, "owner@example.com");
    await requestReset(handler, "owner@example.com");
    await emailsSent();
    // Recording the address is best-effort.
    const record = db.UserAuthVerification.findIndex((row) =>
      String(row.identifier).startsWith("reset-password-mailed-to:"),
    );
    db.UserAuthVerification.splice(record, 1);

    await api.updateUser({ body: { name: "New name" }, headers: session });
    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("owner@example.com", "reset_password"),
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(200);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: false }),
    ]);
  });

  it("survive changes to the account that keep its address", async () => {
    const { handler, api, db } = await createAuthHandler(false);
    const session = await signUpSignedIn(api, "owner@example.com");
    await requestReset(handler, "owner@example.com");
    await emailsSent();

    await api.updateUser({ body: { name: "New name" }, headers: session });
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ name: "New name" }),
    ]);
    const reset = await post(handler, "/reset-password", {
      token: lastResetToken("owner@example.com", "reset_password"),
      newPassword: "a-new-long-enough-password",
    });

    expect(reset.status).toBe(200);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ emailVerified: true }),
    ]);
  });
});

describe("change-email verification emails", () => {
  it.each([false, true])(
    "are not held back by the sign-in cooldown when retried (verification %s)",
    async (requireVerification) => {
      const { api: signUpAPI, db } = await createAuthHandler(false);
      const session = await signUpSignedIn(signUpAPI, "old@example.com");
      db.UserAuthIdentity[0].emailVerified = true;
      const { handler, api } = await createAuthHandler(requireVerification, db);

      // The confirmation goes to the current address; opening it sends the
      // link to the new one. The user doesn't see that one and tries again.
      for (let attempt = 0; attempt < 2; attempt++) {
        sentEmails.length = 0;
        await api.changeEmail({
          body: { newEmail: "new@example.com", callbackURL: "/" },
          headers: session,
        });
        await emailsSent();
        const confirm = sentEmails.find((sent) => sent.type === "change_email");
        await handler(new Request(confirm?.url ?? ""));
        await emailsSent();
        expect(lastVerifyLink("new@example.com")).toContain("/verify-email");
      }
    },
  );

  it("still caps sign-in's own sends", async () => {
    const { handler } = await createAuthHandler(true);
    await signUp(handler, "unverified@example.com");
    await emailsSent();
    sentEmails.length = 0;

    await signIn(handler, "unverified@example.com");
    await emailsSent();

    expect(sentEmails).toEqual([]);
  });
});
