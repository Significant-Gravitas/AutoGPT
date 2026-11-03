// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import { isWaitlistError } from "@/app/api/auth/utils";

// Runs a GitHub sign-in through the real Better Auth handler with this app's
// auth options, as access-blocks-flow.test.ts does for passwords, with
// GitHub's token and profile endpoints answered here.

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

type Handler = (request: Request) => Promise<Response>;

async function createAuthHandler() {
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.stubEnv("AUTH_GITHUB_CLIENT_ID", "github-client-id");
  vi.stubEnv("AUTH_GITHUB_CLIENT_SECRET", "github-client-secret");
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

// GitHub's side of the sign-in: the code exchange and the profile.
function answerAsGitHub(email: string) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === "https://github.com/login/oauth/access_token") {
        return Response.json({
          access_token: "github-access-token",
          token_type: "bearer",
          scope: "user:email",
        });
      }
      if (url === "https://api.github.com/user") {
        return Response.json({ id: 4242, login: "octo", name: "Octo", email });
      }
      if (url === "https://api.github.com/user/emails") {
        return Response.json([{ email, primary: true, verified: true }]);
      }
      throw new Error(`unexpected fetch: ${url}`);
    }),
  );
}

async function signInWithGitHub(handler: Handler) {
  const start = await handler(
    new Request(`${baseURL}/api/auth/sign-in/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({
        provider: "github",
        callbackURL: "/auth/callback",
      }),
    }),
  );
  const { url } = await start.json();
  const state = new URL(url).searchParams.get("state");
  const cookie = start.headers
    .getSetCookie()
    .map((setCookie) => setCookie.split(";")[0])
    .join("; ");
  return handler(
    new Request(
      `${baseURL}/api/auth/callback/github?code=github-code&state=${state}`,
      { headers: { cookie } },
    ),
  );
}

function redirectOf(response: Response) {
  return new URL(response.headers.get("location") ?? "", baseURL);
}

afterEach(() => {
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});

describe("signing up with GitHub", () => {
  it("creates the account and lands on the callback page", async () => {
    const handler = await createAuthHandler();
    answerAsGitHub("octo@example.com");

    const response = await signInWithGitHub(handler);

    expect(response.status).toBe(302);
    expect(redirectOf(response).pathname).toBe("/auth/callback");
    expect(testDB.current.UserAuthIdentity).toEqual([
      expect.objectContaining({ email: "octo@example.com" }),
    ]);
  });

  it("sends an address off the allowlist to the error page with its reason", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "invited@example.com");
    const handler = await createAuthHandler();
    answerAsGitHub("octo@example.com");

    const response = await signInWithGitHub(handler);

    expect(response.status).toBe(302);
    const redirect = redirectOf(response);
    expect(redirect.searchParams.get("error")).toBe("SIGN_UP_NOT_ALLOWED");
    expect(
      isWaitlistError(
        redirect.searchParams.get("error"),
        redirect.searchParams.get("error_description"),
      ),
    ).toBe(true);
    expect(testDB.current.UserAuthIdentity).toEqual([]);
  });
});
