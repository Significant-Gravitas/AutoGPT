// @vitest-environment node
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { GOOGLE_ACCESS_SCOPES } from "@/lib/google-access";
import { MICROSOFT_BASE_SCOPES } from "@/lib/microsoft-setup";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database: the sign-in redirect, then the
// provider's callback, with the provider's endpoints and the backend's
// sign-in connection endpoint answered by MSW. In node rather than happy-dom,
// whose Request drops the state cookie the callback checks.

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
// The JWT plugin isn't loaded here; the token's claims are what matter.
vi.mock("../service-token", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../service-token")>()),
  mintServiceToken: vi.fn(async () => "service-token"),
}));

const baseURL = "http://localhost:3000";
const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";
const IDENTITY = ["openid", "https://www.googleapis.com/auth/userinfo.email"];
// What a connecting Microsoft sign-in asks for (connect-at-sign-in.ts).
const MICROSOFT_CONNECT_SCOPES = ["offline_access", ...MICROSOFT_BASE_SCOPES];
type Handler = (request: Request) => Promise<Response>;

// The clients the backend's connections use: by default the ones sign-in
// uses, so sign-in connects.
async function createAuth(
  connectionClients: Record<string, string | null> = {
    google: "google-client-id",
    microsoft: "microsoft-client-id",
  },
) {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
  vi.stubEnv("BETTER_AUTH_URL", baseURL);
  vi.stubEnv("BETTER_AUTH_SECRET", "test-secret-that-is-at-least-32-chars");
  vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "google-client-id");
  vi.stubEnv("AUTH_GOOGLE_CLIENT_SECRET", "google-client-secret"); // pragma: allowlist secret
  vi.stubEnv("AUTH_MICROSOFT_CLIENT_ID", "microsoft-client-id");
  vi.stubEnv("AUTH_MICROSOFT_CLIENT_SECRET", "microsoft-client-secret"); // pragma: allowlist secret
  server.use(
    http.get("*/api/auth/sign-in-connection", () =>
      HttpResponse.json(connectionClients),
    ),
  );
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
  const { mintServiceToken } = await import("../service-token");
  return { handler: auth.handler as Handler, db, mintServiceToken };
}

afterEach(() => {
  vi.unstubAllEnvs();
});

// What the backend's sign-in connection endpoint was handed.
function backendHandoffs(status = 200) {
  const bodies: Array<Record<string, unknown>> = [];
  server.use(
    http.post("*/api/auth/sign-in-connection", async ({ request }) => {
      bodies.push((await request.json()) as Record<string, unknown>);
      return HttpResponse.json({ result: "created" }, { status });
    }),
  );
  return bodies;
}

function answerAsGoogle(
  granted: string[],
  refreshToken: string | null = "google-refresh",
) {
  server.use(
    http.post("https://oauth2.googleapis.com/token", () =>
      HttpResponse.json({
        token_type: "Bearer",
        access_token: "google-access",
        ...(refreshToken ? { refresh_token: refreshToken } : {}),
        expires_in: 3599,
        scope: granted.join(" "),
        id_token: unsignedJWT({
          sub: "google-account-id",
          email: "pat@gmail.com",
          email_verified: true,
          name: "Pat",
        }),
      }),
    ),
  );
}

function answerAsMicrosoft(granted: string) {
  server.use(
    http.post(
      "https://login.microsoftonline.com/common/oauth2/v2.0/token",
      () =>
        HttpResponse.json({
          token_type: "Bearer",
          access_token: "microsoft-access",
          refresh_token: "microsoft-refresh",
          expires_in: 3600,
          scope: granted,
          id_token: unsignedJWT({
            sub: "pairwise",
            oid: "personal-id",
            tid: PERSONAL_TENANT,
            email: "pat@outlook.com",
            name: "Pat",
          }),
        }),
    ),
  );
}

async function startSignIn(
  handler: Handler,
  provider: string,
  scopes?: readonly string[],
) {
  return handler(
    new Request(`${baseURL}/api/auth/sign-in/social`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Origin: baseURL },
      body: JSON.stringify({
        provider,
        callbackURL: "/auth/callback",
        errorCallbackURL: "/login",
        ...(scopes ? { scopes } : {}),
      }),
    }),
  );
}

// The browser's round trip: start the sign-in, then come back to the
// callback with the provider's code and the state.
async function signInWith(
  handler: Handler,
  provider: string,
  scopes?: readonly string[],
) {
  const start = await startSignIn(handler, provider, scopes);
  const { url } = await start.json();
  const state = new URL(url).searchParams.get("state") ?? "";
  const cookie = start.headers
    .getSetCookie()
    .map((header) => header.split(";")[0])
    .join("; ");
  const callback = await handler(
    new Request(
      `${baseURL}/api/auth/callback/${provider}?code=provider-code&state=${state}`,
      { headers: { cookie } },
    ),
  );
  return {
    authorizeURL: new URL(url),
    landing: callback.headers.get("location"),
  };
}

function unsignedJWT(claims: Record<string, unknown>) {
  const encode = (part: object) =>
    Buffer.from(JSON.stringify(part)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(claims)}.`;
}

describe("Connecting at sign-in", () => {
  it("asks Google for the data scopes and offline access", async () => {
    const { handler } = await createAuth();
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES]);
    backendHandoffs();

    const { authorizeURL } = await signInWith(
      handler,
      "google",
      GOOGLE_ACCESS_SCOPES,
    );

    const asked = authorizeURL.searchParams.get("scope")?.split(" ") ?? [];
    expect(asked).toEqual(expect.arrayContaining([...GOOGLE_ACCESS_SCOPES]));
    expect(authorizeURL.searchParams.get("access_type")).toBe("offline");
  });

  it("hands the backend a Google sign-in's tokens, and never the auth table", async () => {
    const { handler, db, mintServiceToken } = await createAuth();
    const granted = [...IDENTITY, ...GOOGLE_ACCESS_SCOPES];
    answerAsGoogle(granted);
    const handoffs = backendHandoffs();

    const { landing } = await signInWith(
      handler,
      "google",
      GOOGLE_ACCESS_SCOPES,
    );

    expect(landing).toBe("/auth/callback");
    expect(handoffs).toEqual([
      {
        provider: "google",
        access_token: "google-access",
        refresh_token: "google-refresh",
        expires_at: expect.any(Number),
        scope: granted.join(" "),
        id_token: expect.any(String),
      },
    ]);
    // The service token vouches for the user who just signed in.
    const [user] = db.UserAuthIdentity;
    expect(mintServiceToken).toHaveBeenCalledWith(
      "integrations:sign-in-connection",
      { user_id: user.id },
    );
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({
        providerId: "google",
        accessToken: null,
        refreshToken: null,
        idToken: null,
      }),
    ]);
  });

  it("keeps a returning sign-in's tokens out of the auth table too", async () => {
    const { handler, db } = await createAuth();
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES]);
    const handoffs = backendHandoffs();

    await signInWith(handler, "google", GOOGLE_ACCESS_SCOPES);
    // Google issues no refresh token once the scopes are granted.
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES], null);
    await signInWith(handler, "google", GOOGLE_ACCESS_SCOPES);

    expect(handoffs).toHaveLength(2);
    expect(handoffs[1]).toMatchObject({ refresh_token: null });
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({ accessToken: null, refreshToken: null }),
    ]);
  });

  it("connects nothing for a sign-in that asked only who the person is", async () => {
    const { handler, db } = await createAuth();
    // Someone who connected before chose to sign in without connecting:
    // Google's token still carries the data scopes granted then.
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES]);
    const handoffs = backendHandoffs();

    const { landing } = await signInWith(handler, "google");

    expect(landing).toBe("/auth/callback");
    expect(handoffs).toEqual([]);
    expect(db.UserAuthSession).toHaveLength(1);
  });

  it("only signs in where sign-in has a client of its own", async () => {
    // A self-hosted deployment, say, whose Google connections use another
    // client: the backend couldn't renew this one's tokens.
    const { handler, db } = await createAuth({
      google: "connections-client-id",
      microsoft: null,
    });
    // Google still returns the data scopes an earlier sign-in granted.
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES]);
    const handoffs = backendHandoffs();

    const { landing } = await signInWith(handler, "google");

    expect(landing).toBe("/auth/callback");
    expect(handoffs).toEqual([]);
    expect(db.UserAuthSession).toHaveLength(1);
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({ accessToken: null, refreshToken: null }),
    ]);
  });

  it("still signs in when the backend can't keep the connection", async () => {
    const { handler, db } = await createAuth();
    answerAsGoogle([...IDENTITY, ...GOOGLE_ACCESS_SCOPES]);
    const handoffs = backendHandoffs(409);
    const errors = vi.spyOn(console, "error").mockImplementation(() => {});

    const { landing } = await signInWith(
      handler,
      "google",
      GOOGLE_ACCESS_SCOPES,
    );

    expect(handoffs).toHaveLength(1);
    expect(landing).toBe("/auth/callback");
    expect(db.UserAuthSession).toHaveLength(1);
    expect(errors).toHaveBeenCalledWith("A sign-in's connection wasn't kept", {
      provider: "google",
      status: 409,
    });
    errors.mockRestore();
  });

  it("refuses the data scopes where sign-in has a client of its own", async () => {
    const { handler } = await createAuth({
      google: "connections-client-id",
      microsoft: null,
    });

    const start = await startSignIn(handler, "google", GOOGLE_ACCESS_SCOPES);

    expect(start.status).toBe(400);
    expect(await start.json()).toMatchObject({ code: "SCOPES_NOT_ALLOWED" });
  });

  it("refuses any other scope for Google", async () => {
    const { handler } = await createAuth();

    const start = await startSignIn(handler, "google", [
      "https://mail.google.com/",
    ]);

    expect(start.status).toBe(400);
  });

  it("hands the backend a Microsoft sign-in's tokens, and never the auth table", async () => {
    const { handler, db } = await createAuth();
    const handoffs = backendHandoffs();
    answerAsMicrosoft(
      "openid profile email https://graph.microsoft.com/Mail.ReadWrite https://graph.microsoft.com/Files.ReadWrite",
    );

    const { authorizeURL } = await signInWith(
      handler,
      "microsoft",
      MICROSOFT_CONNECT_SCOPES,
    );

    expect(authorizeURL.searchParams.get("scope")).toContain("Mail.ReadWrite");
    expect(handoffs).toEqual([
      expect.objectContaining({
        provider: "microsoft",
        access_token: "microsoft-access",
        refresh_token: "microsoft-refresh",
        id_token: expect.any(String),
      }),
    ]);
    expect(db.UserAuthAccount).toEqual([
      expect.objectContaining({
        providerId: "microsoft",
        accessToken: null,
        refreshToken: null,
        idToken: null,
      }),
    ]);
  });

  it("connects nothing for a Microsoft sign-in granted none of the data scopes", async () => {
    const { handler, db } = await createAuth();
    const handoffs = backendHandoffs();
    // The profile scope and a refresh token, but no mail, files or calendar.
    answerAsMicrosoft(
      "openid profile email https://graph.microsoft.com/User.Read offline_access",
    );

    const { landing } = await signInWith(
      handler,
      "microsoft",
      MICROSOFT_CONNECT_SCOPES,
    );

    expect(landing).toBe("/auth/callback");
    expect(handoffs).toEqual([]);
    expect(db.UserAuthSession).toHaveLength(1);
  });
});
