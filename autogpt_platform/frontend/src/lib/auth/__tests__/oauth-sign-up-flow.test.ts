// @vitest-environment node
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isWaitlistError } from "@/app/api/auth/utils";
import { server } from "@/mocks/mock-server";
import {
  TEAM_EMAIL_REQUIRES_GOOGLE_CODE,
  TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE,
} from "../team-email-policy";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database: a sign-in redirect, then the provider's
// callback, with the provider's own endpoints answered by MSW. In node rather
// than happy-dom, whose Request drops the state cookie the callback checks.

vi.mock("pg", () => ({
  Pool: class {
    async query(sql: string) {
      if (sql.includes("is_auth_access_blocked(")) {
        return { rowCount: 1, rows: [{ blocked: false }] };
      }
      if (sql.includes("record_auth_ip_event(")) {
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
        plugins: options.plugins?.filter((plugin) => plugin.id === "admin"),
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
const PROVIDERS = ["google", "github", "discord"] as const;
type Provider = (typeof PROVIDERS)[number];
type Handler = (request: Request) => Promise<Response>;

async function createAuth(requireVerification = false) {
  vi.stubEnv(
    "AUTH_REQUIRE_EMAIL_VERIFICATION",
    requireVerification ? "true" : "false",
  );
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  for (const provider of PROVIDERS) {
    const prefix = `AUTH_${provider.toUpperCase()}`;
    vi.stubEnv(`${prefix}_CLIENT_ID`, `${provider}-client-id`);
    vi.stubEnv(`${prefix}_CLIENT_SECRET`, `${provider}-client-secret`);
  }
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
  return { handler: auth.handler as Handler, api: auth.api, db };
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("OAuth sign-up", () => {
  it.each(
    PROVIDERS.flatMap((provider) => [
      [provider, false],
      [provider, true],
    ]),
  )(
    "refuses an address %s has not verified (verification %s)",
    async (provider, requireVerification) => {
      const { handler, db } = await createAuth(requireVerification);
      answerAs(provider, { email: "someone@example.com", verified: false });

      const response = await signInWith(handler, provider);

      expect(refusal(response)).toEqual({
        code: "EMAIL_NOT_VERIFIED_BY_PROVIDER",
        message: expect.stringMatching(/verify your email address/),
      });
      expect(db.UserAuthIdentity).toEqual([]);
      expect(db.UserAuthSession).toEqual([]);
    },
  );

  it.each(["github", "discord"] as const)(
    "refuses a team address through %s, even verified",
    async (provider) => {
      const { handler, db } = await createAuth();
      answerAs(provider, { email: "someone@agpt.co", verified: true });

      const response = await signInWith(handler, provider);

      expect(refusal(response)).toEqual({
        code: TEAM_EMAIL_REQUIRES_GOOGLE_CODE,
        message: TEAM_EMAIL_REQUIRES_GOOGLE_MESSAGE,
      });
      expect(db.UserAuthIdentity).toEqual([]);
      expect(db.UserAuthSession).toEqual([]);
    },
  );

  it.each([
    ["google", "someone@agpt.co"],
    ["github", "someone@example.com"],
    ["discord", "someone@example.com"],
  ] as const)("signs %s's verified %s up and in", async (provider, email) => {
    const { handler, db } = await createAuth();
    answerAs(provider, { email, verified: true });

    const response = await signInWith(handler, provider);

    expect(response.status).toBe(302);
    expect(response.headers.get("location")).toBe("/auth/callback");
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ email, emailVerified: true }),
    ]);
    expect(db.UserAuthSession).toHaveLength(1);
  });

  it("sends an address off the allowlist to the error page with its reason", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "invited@example.com");
    const { handler, db } = await createAuth();
    answerAs("github", { email: "someone@example.com", verified: true });

    const { code, message } = refusal(await signInWith(handler, "github"));

    expect(code).toBe("SIGN_UP_NOT_ALLOWED");
    expect(isWaitlistError(code, message)).toBe(true);
    expect(db.UserAuthIdentity).toEqual([]);
  });

  it("still lets an admin create an unverified team account", async () => {
    const { api, db } = await createAuth();

    await api.createUser({
      body: { email: "someone@agpt.co", password: "x".repeat(16), name: "x" },
    });

    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        email: "someone@agpt.co",
        emailVerified: false,
      }),
    ]);
  });
});

// Starts a sign-in as /api/auth/login/with-provider does, then comes back to
// the callback the way the provider redirects the browser.
async function signInWith(handler: Handler, provider: Provider) {
  const start = await handler(
    new Request(`${baseURL}/api/auth/sign-in/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({ provider, callbackURL: "/auth/callback" }),
    }),
  );
  const { url } = await start.json();
  const state = new URL(url).searchParams.get("state") ?? "";
  const cookie = start.headers
    .getSetCookie()
    .map((header) => header.split(";")[0])
    .join("; ");

  return handler(
    new Request(
      `${baseURL}/api/auth/callback/${provider}?code=provider-code&state=${state}`,
      { headers: { cookie } },
    ),
  );
}

// A refused sign-up lands on Better Auth's error page, carrying the hook's code
// and message.
function refusal(response: Response) {
  expect(response.status).toBe(302);
  const location = new URL(response.headers.get("location") ?? "");
  expect(location.pathname).toBe("/api/auth/error");
  return {
    code: location.searchParams.get("error"),
    message: location.searchParams.get("error_description"),
  };
}

// What each provider's token and profile endpoints answer for this person.
function answerAs(
  provider: Provider,
  { email, verified }: { email: string; verified: boolean },
) {
  const tokens = { access_token: "provider-token", token_type: "Bearer" };
  if (provider === "google") {
    const idToken = unsignedJWT({
      sub: "google-account-id",
      email,
      email_verified: verified,
      name: "Someone",
    });
    server.use(
      http.post("https://oauth2.googleapis.com/token", () =>
        HttpResponse.json({ ...tokens, id_token: idToken }),
      ),
    );
  } else if (provider === "github") {
    server.use(
      http.post("https://github.com/login/oauth/access_token", () =>
        HttpResponse.json(tokens),
      ),
      http.get("https://api.github.com/user", () =>
        HttpResponse.json({ id: 1, login: "someone", email, avatar_url: "" }),
      ),
      http.get("https://api.github.com/user/emails", () =>
        HttpResponse.json([{ email, primary: true, verified }]),
      ),
    );
  } else {
    server.use(
      http.post("https://discord.com/api/oauth2/token", () =>
        HttpResponse.json(tokens),
      ),
      // Better Fetch sends `@me` percent-encoded.
      http.get("https://discord.com/api/users/%40me", () =>
        HttpResponse.json({
          id: "80351110224678912",
          username: "someone",
          email,
          verified,
          avatar: null,
          discriminator: "0",
        }),
      ),
    );
  }
}

// The callback only decodes Google's ID token, so it needs no signature.
function unsignedJWT(claims: Record<string, unknown>) {
  const encode = (part: object) =>
    Buffer.from(JSON.stringify(part)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(claims)}.`;
}
