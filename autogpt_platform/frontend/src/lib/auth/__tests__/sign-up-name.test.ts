// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { getGreetingName } from "@/app/(platform)/copilot/components/EmptySession/helpers";
import { mapSessionUser } from "../types";

// The name a new account gets, from the sign-up form's "What may we call you?"
// or from the Google profile, through the real sign-up action and Better Auth
// (this app's options, an in-memory database) to the greeting and the platform
// `User` row the emails read.

vi.mock("pg", () => ({
  Pool: class {
    async query() {
      return { rows: [], rowCount: 0 };
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
vi.mock("next/headers", () => ({
  headers: vi.fn(async () => new Headers()),
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
vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));

// The platform `User` rows the auth hook writes, by the real insert's rule.
const platformUsers = vi.hoisted(
  () => new Map<string, { email: string; name: string | null }>(),
);
vi.mock("../provision-platform-user", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("../provision-platform-user")>();
  return {
    ...actual,
    provisionPlatformUser: (
      _pool: unknown,
      user: Parameters<typeof actual.provisionPlatformUser>[1],
    ) =>
      actual.provisionPlatformUser(
        {
          query: vi.fn().mockImplementation(async (_sql, params) => {
            const [id, email, name] = params as [string, string, string | null];
            platformUsers.set(id, { email, name });
            return { rowCount: 1 };
          }),
        },
        user,
      ),
  };
});

vi.mock("@/app/api/__generated__/endpoints/auth/auth", () => ({
  postV1GetOrCreateUser: vi.fn(async () => ({
    status: 200,
    data: {},
    headers: new Headers({ "X-AutoGPT-User-Created": "false" }),
  })),
}));
vi.mock("@/app/api/helpers", () => ({
  getOnboardingStatus: vi.fn(async () => ({ shouldShowOnboarding: true })),
}));

const baseURL = "http://localhost:3000";
const password = "a-long-enough-password"; // pragma: allowlist secret

beforeEach(() => {
  platformUsers.clear();
});

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("an email sign-up", () => {
  it.each([
    ["the name given", "sam.carter@example.com", "  Sam  ", "Sam"],
    // "john" at john@ is a name the user chose, not the address read back.
    ["a name that matches the address", "john@example.com", "john", "john"],
  ])("keeps %s and is greeted by it", async (_case, email, given, expected) => {
    const { auth, signup } = await createAuth();

    const result = await signup(email, password, password, false, null, given);

    expect(result).toEqual({ success: true, next: "/onboarding" });
    const user = await sessionUser(auth, email);
    expect(user.user_metadata.name).toBe(expected);
    expect(getGreetingName(user)).toBe(expected);
    expect(platformUsers.get(user.id)?.name).toBe(expected);
  });

  it.each([undefined, "   "])(
    'without a name (%j) stores none and is greeted as "there"',
    async (given) => {
      const { auth, signup } = await createAuth();

      await signup(
        "jane.doe87@example.com",
        password,
        password,
        false,
        null,
        given,
      );

      const user = await sessionUser(auth, "jane.doe87@example.com");
      expect(user.user_metadata.name).toBe("");
      expect(getGreetingName(user)).toBeNull();
      expect(platformUsers.get(user.id)?.name).toBeNull();
    },
  );
});

describe("a Google sign-up", () => {
  it.each([
    ["sam.carter@gmail.com", "Sam Carter", "Sam"],
    ["john@gmail.com", "john", "john"],
  ])(
    "as %s takes the Google profile's name %j",
    async (email, googleName, greeting) => {
      vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "google-client-id");
      vi.stubEnv("AUTH_GOOGLE_CLIENT_SECRET", "google-client-secret");
      const { auth } = await createAuth();

      const landing = await signInWithGoogle(auth, { email, name: googleName });

      expect(landing.status).toBe(302);
      expect(landing.headers.get("location")).toBe("/auth/callback");
      const [identity] = testDB.current.UserAuthIdentity as Array<{
        id: string;
        name: string;
      }>;
      expect(identity.name).toBe(googleName);
      expect(platformUsers.get(identity.id)?.name).toBe(googleName);
      expect(getGreetingName(mapSessionUser({ ...identity, email }))).toBe(
        greeting,
      );
    },
  );
});

type Auth = Awaited<ReturnType<typeof createAuth>>["auth"];

async function createAuth() {
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
  const { signup } = await import("@/app/(no-navbar)/signup/actions");
  return { auth, signup };
}

// What /api/auth/user hands the app for this account once it signs in.
async function sessionUser(auth: Auth, email: string) {
  const { headers } = await auth.api.signInEmail({
    body: { email, password },
    returnHeaders: true,
  });
  const cookie = headers.get("set-cookie")?.split(";")[0] ?? "";
  const session = await auth.api.getSession({
    headers: new Headers({ cookie }),
  });
  if (!session) throw new Error("No session after signing in");
  return mapSessionUser(session.user);
}

// Google's side of the redirect: the token endpoint answers with an ID token
// carrying the profile, which Better Auth reads the account from.
async function signInWithGoogle(
  auth: Auth,
  profile: { email: string; name: string },
) {
  const start = await auth.handler(
    new Request(`${baseURL}/api/auth/sign-in/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({
        provider: "google",
        callbackURL: "/auth/callback",
      }),
    }),
  );
  const { url } = (await start.json()) as { url: string };
  const state = new URL(url).searchParams.get("state");
  const cookie = start.headers
    .getSetCookie()
    .map((value) => value.split(";")[0])
    .join("; ");

  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      Response.json({
        access_token: "google-access-token",
        token_type: "Bearer",
        expires_in: 3600,
        id_token: unsignedJWT({
          sub: "google-account-id",
          email: profile.email,
          email_verified: true,
          name: profile.name,
        }),
      }),
    ),
  );
  return auth.handler(
    new Request(
      `${baseURL}/api/auth/callback/google?code=google-code&state=${state}`,
      { headers: { cookie } },
    ),
  );
}

function unsignedJWT(payload: Record<string, unknown>) {
  const encode = (part: object) =>
    Buffer.from(JSON.stringify(part)).toString("base64url");
  return `${encode({ alg: "RS256", kid: "test" })}.${encode(payload)}.signature`;
}
