// @vitest-environment node
import { SignJWT } from "jose";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getEmailVerificationCallbackURL } from "../email-verification";
import {
  holdVerifyLink,
  readHeldLink,
  verifyHeldLink,
  VERIFY_EMAIL_CONFIRM_PATH,
} from "../verify-email-confirm";

// The app's /api/auth GET route over the real Better Auth handler, on an
// in-memory database: opening the emailed link must change nothing, and only
// the confirm page's click verifies and signs in. Node, not happy-dom, so the
// route's Set-Cookie survives.

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
vi.mock("better-auth/plugins", () => ({
  admin: vi.fn(() => ({ id: "admin" })),
  bearer: vi.fn(() => ({ id: "bearer", hooks: { before: [], after: [] } })),
  jwt: vi.fn(() => ({ id: "jwt" })),
}));
vi.mock("../supabase-bridge", () => ({
  supabaseBridge: vi.fn(() => ({ id: "supabase-bridge" })),
}));
vi.mock("../email-mirror", () => ({
  mirrorVerifiedEmailToPlatformUser: vi.fn(),
}));
vi.mock("../provision-platform-user", () => ({
  provisionPlatformUser: vi.fn(),
  platformUserExists: vi.fn(async () => false),
}));
vi.mock("../ban-record", () => ({ assertNotBannedAddress: vi.fn() }));
const sentEmails = vi.hoisted(
  () => [] as Array<{ to: string; type: string; url: string }>,
);
vi.mock("../email", () => ({
  sendAuthEmail: vi.fn(async (email: (typeof sentEmails)[number]) => {
    sentEmails.push(email);
  }),
}));

const baseURL = "http://localhost:3000";
const secret = "test-secret-that-is-at-least-32-chars"; // pragma: allowlist secret
const password = "a-long-enough-password"; // pragma: allowlist secret
const email = "new@example.com";
const callbackURL = getEmailVerificationCallbackURL({
  next: "/library",
  marketingOptOut: true,
});

async function createApp() {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "true");
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", secret);
  vi.doUnmock("@/lib/auth/auth");
  vi.resetModules();
  const db = {
    UserAuthIdentity: [] as Array<Record<string, unknown>>,
    UserAuthSession: [] as Array<Record<string, unknown>>,
    UserAuthAccount: [] as Array<Record<string, unknown>>,
    UserAuthVerification: [] as Array<Record<string, unknown>>,
  };
  testDB.current = db;
  const { auth } = await import("../auth");
  const { GET } = await import("@/app/api/auth/[...all]/route");
  await auth.api.signUpEmail({
    body: { email, password, name: "new", callbackURL },
  });
  while (pendingAfterResponse.length) {
    await Promise.all(pendingAfterResponse.splice(0));
  }
  const verifyLink = sentEmails.find((sent) => sent.to === email)?.url ?? "";
  return { api: auth.api, GET, db, verifyLink };
}

// What the browser sends the confirm page back, from the route's Set-Cookie.
function heldLinkFrom(response: Response) {
  const [pair] = response.headers.getSetCookie()[0].split(";");
  const [name, value] = pair.split("=");
  const link = readHeldLink({
    get: (key) =>
      key === name ? { value: decodeURIComponent(value) } : undefined,
  });
  if (!link) throw new Error("The route held no link");
  return link;
}

function signedToken(payload: Record<string, unknown>, expiresAt: number) {
  return new SignJWT(payload)
    .setProtectedHeader({ alg: "HS256" })
    .setExpirationTime(expiresAt)
    .sign(new TextEncoder().encode(secret));
}

beforeEach(() => {
  sentEmails.length = 0;
  pendingAfterResponse.length = 0;
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("opening an emailed verification link", () => {
  it("shows the confirm page and neither verifies the address nor signs in", async () => {
    const { GET, db, verifyLink } = await createApp();

    const response = await GET(new Request(verifyLink));

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe(VERIFY_EMAIL_CONFIRM_PATH);
    const cookie = response.headers.getSetCookie()[0];
    expect(cookie).toContain("HttpOnly");
    expect(cookie).toContain(`Path=${VERIFY_EMAIL_CONFIRM_PATH}`);
    expect(db.UserAuthSession).toEqual([]);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ email, emailVerified: false }),
    ]);
  });

  it("still lets the user's click verify, sign in and land on /auth/callback after an earlier open", async () => {
    const { api, GET, db, verifyLink } = await createApp();
    await GET(new Request(verifyLink));

    const held = heldLinkFrom(await GET(new Request(verifyLink)));
    const target = await verifyHeldLink(api, held, new Headers());

    expect(target).toBe(callbackURL);
    expect(db.UserAuthSession).toHaveLength(1);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ email, emailVerified: true }),
    ]);
  });

  it("signs no one in on a second click, which /auth/callback answers with the verified notice", async () => {
    const { api, GET, db, verifyLink } = await createApp();
    const held = heldLinkFrom(await GET(new Request(verifyLink)));
    await verifyHeldLink(api, held, new Headers());

    const target = await verifyHeldLink(api, held, new Headers());

    expect(target).toBe(callbackURL);
    expect(db.UserAuthSession).toHaveLength(1);
  });

  it.each([
    ["an expired", "TOKEN_EXPIRED"],
    ["a tampered", "INVALID_TOKEN"],
  ])(
    "sends %s link's click to /auth/callback with its error",
    async (_kind, code) => {
      const { api, db } = await createApp();
      const userId = String(db.UserAuthIdentity[0].id);
      const token =
        code === "TOKEN_EXPIRED"
          ? await signedToken({ email, userId }, 1)
          : "not-a-token";

      const target = await verifyHeldLink(
        api,
        { token, callbackURL },
        new Headers(),
      );

      expect(target).toBe(`${callbackURL}&error=${code}`);
      expect(db.UserAuthSession).toEqual([]);
    },
  );
});

describe("links the route leaves to Better Auth", () => {
  it.each([
    ["a change of address", "true", { updateTo: "next@example.com" }],
    ["a sign-up link with verification off", "false", {}],
  ])("%s", async (_kind, requireVerification, extra) => {
    vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", requireVerification);
    const token = await signedToken(
      { email, userId: "u-1", ...extra },
      Math.floor(Date.now() / 1000) + 60,
    );

    const held = holdVerifyLink(
      new Request(`${baseURL}/api/auth/verify-email?token=${token}`),
    );

    expect(held).toBeNull();
  });
});
