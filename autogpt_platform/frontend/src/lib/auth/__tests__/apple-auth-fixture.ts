import { randomUUID } from "node:crypto";
import { exportJWK, generateKeyPair, SignJWT, type JWTPayload } from "jose";
import { vi } from "vitest";

const appleTestState = vi.hoisted(() => ({
  database: {} as Record<string, unknown[]>,
  blocked: false,
  provisioned: vi.fn(),
}));

export function getAppleTestState() {
  return appleTestState;
}

vi.mock("pg", () => ({
  Pool: class {
    async query(sql: string) {
      if (sql.includes("is_auth_access_blocked(")) {
        return { rowCount: 1, rows: [{ blocked: appleTestState.blocked }] };
      }
      if (sql.includes("record_auth_ip_event(")) {
        return { rowCount: 1, rows: [] };
      }
      throw new Error("Unexpected database query in Apple sign-in test");
    }
  },
}));
vi.mock("node:dns/promises", () => ({
  Resolver: class {
    async resolveMx() {
      return [{ exchange: "mail.example.test" }];
    }
    async resolve4() {
      return ["192.0.2.1"];
    }
    async resolve6() {
      return [];
    }
  },
}));
vi.mock("better-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth")>();
  const { memoryAdapter } = await import("better-auth/adapters/memory");
  return {
    ...actual,
    betterAuth: (options: Parameters<typeof actual.betterAuth>[0]) =>
      actual.betterAuth({
        ...options,
        advanced: {
          ...options.advanced,
          disableOriginCheck: false,
          disableCSRFCheck: false,
        },
        database: memoryAdapter(appleTestState.database),
        rateLimit: { enabled: false },
      }),
  };
});
vi.mock("better-auth/next-js", () => ({
  nextCookies: () => ({ id: "next-cookies" }),
}));
vi.mock("next/server", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/server")>()),
  after: vi.fn(),
}));
vi.mock("../supabase-bridge", () => ({
  supabaseBridge: () => ({ id: "supabase-bridge" }),
}));
vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));
vi.mock("../email-mirror", () => ({
  mirrorVerifiedEmailToPlatformUser: vi.fn(),
}));
vi.mock("../provision-platform-user", () => ({
  provisionPlatformUser: appleTestState.provisioned,
  platformUserExists: vi.fn(async () => true),
}));

export const origin = "https://platform.example.test";
export const bundleID = "com.autogpt.test";
export const nativeHeaders = { "X-AutoGPT-Native": "1" };
export const nonce = "test-apple-authorization-nonce";
export type AuthHandler = (request: Request) => Promise<Response>;

let appleKey: CryptoKey;

export async function prepareAppleAuth() {
  vi.stubEnv("BETTER_AUTH_URL", origin);
  vi.stubEnv("BETTER_AUTH_SECRET", randomUUID() + randomUUID());
  vi.stubEnv("AUTH_APPLE_APP_BUNDLE_IDENTIFIER", bundleID);
  vi.stubEnv("AUTH_APPLE_CLIENT_ID", "");
  vi.stubEnv("AUTH_APPLE_TEAM_ID", "");
  vi.stubEnv("AUTH_APPLE_KEY_ID", "");
  vi.stubEnv("AUTH_APPLE_PRIVATE_KEY", "");
  vi.stubEnv("AUTH_ALLOW_NEW_ACCOUNTS", "true");
  vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "");
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
  appleTestState.database = {
    UserAuthIdentity: [],
    UserAuthSession: [],
    UserAuthAccount: [],
    UserAuthVerification: [],
    UserAuthJwks: [],
  };
  appleTestState.blocked = false;
  appleTestState.provisioned.mockReset().mockResolvedValue("created");
  const keys = await generateKeyPair("RS256");
  appleKey = keys.privateKey;
  const jwk = await exportJWK(keys.publicKey);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = input instanceof Request ? input.url : String(input);
      if (url === "https://appleid.apple.com/auth/keys") {
        return Response.json({
          keys: [{ ...jwk, kid: "apple-test-key", alg: "RS256", use: "sig" }],
        });
      }
      throw new Error("Unexpected external request in Apple sign-in test");
    }),
  );
  vi.doUnmock("../auth");
  vi.resetModules();
}

export async function loadAppleAuth() {
  const { auth } = await import("../auth");
  return auth.handler as AuthHandler;
}

export function appleIdentity(claims: JWTPayload = {}) {
  const now = Math.floor(Date.now() / 1000);
  return new SignJWT({
    iss: "https://appleid.apple.com",
    aud: bundleID,
    sub: "apple-user-123",
    iat: now,
    exp: now + 300,
    email: "apple-user@privaterelay.appleid.com",
    email_verified: "true",
    nonce,
    ...claims,
  })
    .setProtectedHeader({ alg: "RS256", kid: "apple-test-key" })
    .sign(appleKey);
}

export function authRequest(
  endpoint: string,
  body?: unknown,
  headers: Record<string, string> = nativeHeaders,
) {
  return new Request(`${origin}/api/auth/${endpoint}`, {
    method: body === undefined ? "GET" : "POST",
    headers: { "Content-Type": "application/json", ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
}

export async function signInApple(
  handler: AuthHandler,
  claims: JWTPayload = {},
  options: { nonce?: string | null; headers?: Record<string, string> } = {},
) {
  return handler(
    authRequest(
      "sign-in/social",
      {
        provider: "apple",
        idToken: {
          token: await appleIdentity(claims),
          nonce: options.nonce === null ? undefined : (options.nonce ?? nonce),
          user: { name: { firstName: "Apple", lastName: "User" } },
        },
      },
      options.headers,
    ),
  );
}
