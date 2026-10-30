// @vitest-environment node
// The browser environment's Request and Response drop Cookie and Set-Cookie,
// which carry the OAuth state from one leg of the sign-in to the other.
import { exportJWK, generateKeyPair, SignJWT } from "jose";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { isWaitlistSignInError } from "../microsoft-login";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database, with Microsoft's token endpoint answered
// here, so "Continue with Microsoft" is checked from the redirect that starts
// it to the one it ends on.

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
  platformUserExists: vi.fn(async () => false),
}));
vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));
// Sign-up looks the address's domain up in DNS; these addresses are made up.
vi.mock("../email-deliverability", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../email-deliverability")>()),
  assertEmailHasMailServer: vi.fn(),
}));
vi.mock("../disposable-email", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../disposable-email")>()),
  assertNotDisposableEmail: vi.fn(),
}));

const baseURL = "http://localhost:3000";
const callbackURL = `${baseURL}/auth/callback?provider=microsoft`;
const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";

type Handler = (request: Request) => Promise<Response>;

function emptyDB() {
  return {
    UserAuthIdentity: [] as Array<Record<string, unknown>>,
    UserAuthSession: [] as Array<Record<string, unknown>>,
    UserAuthAccount: [] as Array<Record<string, unknown>>,
    UserAuthVerification: [] as Array<Record<string, unknown>>,
  };
}

async function createAuthHandler(requireVerification = true) {
  vi.stubEnv(
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    requireVerification ? "true" : "false",
  );
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.stubEnv("AUTH_MICROSOFT_CLIENT_ID", "microsoft-client-id");
  vi.stubEnv("AUTH_MICROSOFT_CLIENT_SECRET", "microsoft-client-secret"); // pragma: allowlist secret
  vi.doUnmock("../auth");
  vi.resetModules();

  const db = emptyDB();
  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler as Handler, db };
}

type TestDB = Awaited<ReturnType<typeof createAuthHandler>>["db"];

// An AutoGPT account that already uses the address, verified as a password
// or Google sign-up leaves it.
function existingAccount(db: TestDB, email: string) {
  const now = new Date();
  db.UserAuthIdentity.push({
    id: "existing-user",
    name: "Pat",
    email,
    emailVerified: true,
    image: null,
    createdAt: now,
    updatedAt: now,
  });
}

// Microsoft's ID token, as its token endpoint returns it. Better Auth reads
// the claims without checking the signature: it came straight from Microsoft
// over TLS, in exchange for the code.
function idToken(claims: Record<string, unknown>) {
  const part = (value: object) =>
    Buffer.from(JSON.stringify(value)).toString("base64url");
  return `${part({ alg: "RS256", typ: "JWT" })}.${part({ name: "Pat", ...claims })}.signature`;
}

function cookiesFrom(response: Response) {
  return response.headers
    .getSetCookie()
    .map((cookie) => cookie.split(";")[0])
    .join("; ");
}

// The browser's whole round trip: start the sign-in, let Microsoft answer for
// `claims`, and come back to the callback with the code and state.
async function continueWithMicrosoft(
  handler: Handler,
  claims: Record<string, unknown>,
) {
  server.use(
    http.post(
      "https://login.microsoftonline.com/common/oauth2/v2.0/token",
      () =>
        HttpResponse.json({
          token_type: "Bearer",
          access_token: "microsoft-access-token",
          refresh_token: "microsoft-refresh-token",
          expires_in: 3600,
          scope: "openid profile email User.Read",
          id_token: idToken(claims),
        }),
    ),
    http.get(
      "https://graph.microsoft.com/v1.0/me/photos/*",
      () => new HttpResponse(null, { status: 404 }),
    ),
  );
  const start = await handler(
    new Request(`${baseURL}/api/auth/sign-in/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({
        provider: "microsoft",
        callbackURL,
        errorCallbackURL: "/login",
      }),
    }),
  );
  const { url } = await start.json();
  const state = new URL(url).searchParams.get("state") ?? "";
  const callback = await handler(
    new Request(
      `${baseURL}/api/auth/callback/microsoft?${new URLSearchParams({ code: "code", state })}`,
      { headers: { cookie: cookiesFrom(start) } },
    ),
  );
  return callback.headers.get("location");
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("Continue with Microsoft", () => {
  it("signs up a personal account on its own Outlook.com address", async () => {
    const { handler, db } = await createAuthHandler();

    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "personal-id",
      tid: PERSONAL_TENANT,
      email: "pat@outlook.com",
    });

    expect(landing).toBe(callbackURL);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        email: "pat@outlook.com",
        emailVerified: true,
      }),
    ]);
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({
        providerId: "microsoft",
        accountId: `${PERSONAL_TENANT}:personal-id`,
      }),
    ]);
    expect(db.UserAuthSession).toHaveLength(1);
  });

  it("signs a work address its organization verified in to the account that uses it", async () => {
    const { handler, db } = await createAuthHandler();
    existingAccount(db, "pat@company.example");

    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "work-id",
      tid: "company-tenant",
      email: "pat@company.example",
      xms_edov: true,
    });

    expect(landing).toBe(callbackURL);
    expect(db.UserAuthIdentity).toHaveLength(1);
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({
        providerId: "microsoft",
        accountId: "company-tenant:work-id",
        userId: "existing-user",
      }),
    ]);
    expect(db.UserAuthSession).toEqual([
      expect.objectContaining({ userId: "existing-user" }),
    ]);
  });

  it.each([
    [
      "a personal account's other address",
      { tid: PERSONAL_TENANT, email: "pat@company.example" },
    ],
    [
      "a work address its organization never verified",
      { tid: "another-tenant", email: "pat@company.example" },
    ],
    [
      "a personal account invited into another organization as a guest",
      {
        tid: "attacker-tenant",
        idp: "live.com",
        email: "pat@company.example",
        xms_edov: true,
      },
    ],
  ])("keeps %s out of the account that uses it", async (_case, claims) => {
    const { handler, db } = await createAuthHandler();
    existingAccount(db, "pat@company.example");

    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "someone-else",
      ...claims,
    });

    expect(landing).toBe("/login?error=account_not_linked");
    expect(db.UserAuthAccount).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it.each([true, false])(
    "creates no account for an address Microsoft didn't confirm (verification required: %s)",
    async (requireVerification) => {
      const { handler, db } = await createAuthHandler(requireVerification);

      const landing = await continueWithMicrosoft(handler, {
        sub: "pairwise",
        oid: "personal-id",
        tid: PERSONAL_TENANT,
        email: "pat@gmail.example",
      });

      expect(landing).toBe("/login?error=unconfirmed_email");
      expect(db.UserAuthIdentity).toEqual([]);
      expect(db.UserAuthAccount).toEqual([]);
      expect(db.UserAuthSession).toEqual([]);
    },
  );

  it("keeps an AutoGPT team address to Google sign-up", async () => {
    const { handler, db } = await createAuthHandler();

    // Only the team's own tenant can confirm an @agpt.co address.
    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "work-id",
      tid: "agpt-tenant",
      email: "someone@agpt.co",
      xms_edov: true,
    });

    expect(landing).toBe("/login?error=team_email_requires_google");
    expect(db.UserAuthIdentity).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("still signs a team member's existing account in with Microsoft", async () => {
    const { handler, db } = await createAuthHandler();
    existingAccount(db, "pat@agpt.co");

    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "work-id",
      tid: "agpt-tenant",
      email: "pat@agpt.co",
      xms_edov: true,
    });

    expect(landing).toBe(callbackURL);
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({
        providerId: "microsoft",
        userId: "existing-user",
      }),
    ]);
  });

  it("refuses a sign-in that hands over a Microsoft ID token directly", async () => {
    const { handler, db } = await createAuthHandler();
    // A token signed with the key served as Microsoft's, for an address
    // Microsoft vouches for: only the switch that turns ID-token sign-in off
    // can refuse it.
    const keys = await generateKeyPair("RS256");
    const jwk = await exportJWK(keys.publicKey);
    server.use(
      http.get(
        "https://login.microsoftonline.com/common/discovery/v2.0/keys",
        () =>
          HttpResponse.json({
            keys: [{ ...jwk, kid: "signing-key", alg: "RS256", use: "sig" }],
          }),
      ),
      http.get(
        "https://graph.microsoft.com/v1.0/me/photos/*",
        () => new HttpResponse(null, { status: 404 }),
      ),
    );
    const token = await new SignJWT({
      tid: PERSONAL_TENANT,
      oid: "personal-id",
      email: "pat@outlook.com",
      name: "Pat",
    })
      .setProtectedHeader({ alg: "RS256", kid: "signing-key" })
      .setIssuer(`https://login.microsoftonline.com/${PERSONAL_TENANT}/v2.0`)
      .setAudience("microsoft-client-id")
      .setSubject("pairwise")
      .setIssuedAt()
      .setExpirationTime("10m")
      .sign(keys.privateKey);

    const response = await handler(
      new Request(`${baseURL}/api/auth/sign-in/social`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Origin: baseURL },
        body: JSON.stringify({ provider: "microsoft", idToken: { token } }),
      }),
    );

    expect(response.status).toBe(401);
    expect(db.UserAuthIdentity).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("sends a sign-up the waitlist refuses back as one /login recognizes", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "@allowed.example");
    const { handler, db } = await createAuthHandler();

    const landing = await continueWithMicrosoft(handler, {
      sub: "pairwise",
      oid: "personal-id",
      tid: PERSONAL_TENANT,
      email: "pat@outlook.com",
    });

    const code = new URL(landing ?? "", baseURL).searchParams.get("error");
    expect(isWaitlistSignInError(code)).toBe(true);
    expect(db.UserAuthIdentity).toEqual([]);
  });
});
