// @vitest-environment node
// The browser environment's Request and Response drop Cookie and Set-Cookie,
// which carry the OAuth state and the code's sign-in from one request to the
// next.
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { isWaitlistError } from "@/app/api/auth/utils";
import { server } from "@/mocks/mock-server";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database, the providers' own endpoints answered by
// MSW: a provider sign-in that can't vouch for its address comes back to
// /login waiting on a code, and the code's endpoint finishes it.

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
        plugins: options.plugins?.filter(
          (plugin) => plugin.id === "admin" || plugin.id === "sign-in-code",
        ),
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
const sendAuthEmail = vi.hoisted(() => vi.fn());
vi.mock("../email", () => ({ sendAuthEmail }));
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
const PERSONAL_TENANT = "9188040d-6c67-4c5b-b112-36a304b66dad";
const PROVIDERS = ["google", "github", "discord", "microsoft"] as const;
type Provider = (typeof PROVIDERS)[number];
type Handler = (request: Request) => Promise<Response>;
type Row = Record<string, unknown>;

async function createAuth() {
  vi.stubEnv("AUTH_REQUIRE_EMAIL_VERIFICATION", "false");
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
    UserAuthIdentity: [] as Row[],
    UserAuthSession: [] as Row[],
    UserAuthAccount: [] as Row[],
    UserAuthVerification: [] as Row[],
  };
  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler as Handler, db };
}

type TestDB = Awaited<ReturnType<typeof createAuth>>["db"];

afterEach(() => {
  vi.unstubAllEnvs();
  sendAuthEmail.mockReset();
});

// One browser: the cookies each response sets, sent with the next request.
class Browser {
  private jar = new Map<string, string>();

  constructor(private handler: Handler) {}

  async request(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers);
    headers.set("cookie", this.cookieHeader());
    if (init.method === "POST") {
      headers.set("Origin", baseURL);
      headers.set("Content-Type", "application/json");
    }
    const response = await this.handler(
      new Request(`${baseURL}${path}`, { ...init, headers }),
    );
    for (const header of response.headers.getSetCookie()) {
      const [pair] = header.split(";");
      const [name, ...value] = pair.split("=");
      const expired = /max-age=0/i.test(header) || value.join("=") === "";
      if (expired) this.jar.delete(name);
      else this.jar.set(name, value.join("="));
    }
    return response;
  }

  has(cookie: string) {
    return [...this.jar.keys()].some((name) => name.endsWith(cookie));
  }

  private cookieHeader() {
    return [...this.jar].map(([name, value]) => `${name}=${value}`).join("; ");
  }

  // The provider's whole round trip, as /api/auth/login/with-provider starts
  // it: where the callback sends the browser.
  async signInWith(provider: Provider) {
    const start = await this.request("/api/auth/sign-in/social", {
      method: "POST",
      body: JSON.stringify({
        provider,
        callbackURL: "/auth/callback",
        errorCallbackURL: "/login",
      }),
    });
    const { url } = await start.json();
    const state = new URL(url).searchParams.get("state") ?? "";
    const callback = await this.request(
      `/api/auth/callback/${provider}?code=provider-code&state=${state}`,
    );
    expect(callback.status).toBe(302);
    return new URL(callback.headers.get("location") ?? "", baseURL);
  }

  enterCode(code: string) {
    return this.request("/api/auth/sign-in-code/verify", {
      method: "POST",
      body: JSON.stringify({ code }),
    });
  }

  resendCode() {
    return this.request("/api/auth/sign-in-code/resend", {
      method: "POST",
      body: "{}",
    });
  }

  async pending() {
    const response = await this.request("/api/auth/sign-in-code/status");
    return (await response.json()).pending;
  }
}

interface Person {
  email: string;
  verified?: boolean;
  // Google: the Workspace domain. Microsoft: the claims besides the address.
  hd?: string;
  microsoft?: Record<string, unknown>;
}

// What each provider's token and profile endpoints answer for this person.
function answerAs(provider: Provider, person: Person) {
  const { email, verified = true } = person;
  const tokens = { access_token: "provider-token", token_type: "Bearer" };
  if (provider === "google") {
    const idToken = unsignedJWT({
      sub: "google-account-id",
      email,
      email_verified: verified,
      name: "Pat",
      ...(person.hd ? { hd: person.hd } : {}),
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
        HttpResponse.json({ id: 1, login: "pat", email, avatar_url: "" }),
      ),
      http.get("https://api.github.com/user/emails", () =>
        HttpResponse.json([{ email, primary: true, verified }]),
      ),
    );
  } else if (provider === "discord") {
    server.use(
      http.post("https://discord.com/api/oauth2/token", () =>
        HttpResponse.json(tokens),
      ),
      // Better Fetch sends `@me` percent-encoded.
      http.get("https://discord.com/api/users/%40me", () =>
        HttpResponse.json({
          id: "80351110224678912",
          username: "pat",
          email,
          verified,
          avatar: null,
          discriminator: "0",
        }),
      ),
    );
  } else {
    server.use(
      http.post(
        "https://login.microsoftonline.com/common/oauth2/v2.0/token",
        () =>
          HttpResponse.json({
            ...tokens,
            expires_in: 3600,
            scope: "openid profile email",
            id_token: unsignedJWT({
              sub: "pairwise",
              oid: "microsoft-account-id",
              name: "Pat",
              email,
              ...person.microsoft,
            }),
          }),
      ),
    );
  }
}

// The callback only decodes the providers' ID tokens, so they need no
// signature.
function unsignedJWT(claims: Record<string, unknown>) {
  const encode = (part: object) =>
    Buffer.from(JSON.stringify(part)).toString("base64url");
  return `${encode({ alg: "none" })}.${encode(claims)}.`;
}

// An AutoGPT account that already uses the address, verified as a password
// sign-up leaves it.
function existingAccount(db: TestDB, email: string, emailVerified = true) {
  const now = new Date();
  db.UserAuthIdentity.push({
    id: "existing-user",
    name: "Pat",
    email,
    emailVerified,
    image: null,
    createdAt: now,
    updatedAt: now,
  });
  db.UserAuthAccount.push({
    id: "existing-password",
    accountId: "existing-user",
    providerId: "credential",
    userId: "existing-user",
    password: "not-a-real-hash", // pragma: allowlist secret
    createdAt: now,
    updatedAt: now,
  });
}

function emailsOfType(type: string) {
  return sendAuthEmail.mock.calls
    .map(([args]) => args)
    .filter((args) => args.type === type);
}

// The last code emailed.
function emailedCode() {
  const codes = emailsOfType("sign_in_code");
  expect(codes.length).toBeGreaterThan(0);
  return codes[codes.length - 1].code as string;
}

function wrong(code: string) {
  return code === "000000" ? "111111" : "000000";
}

function linkedAccounts(db: TestDB, providerId: string) {
  return db.UserAuthAccount.filter((row) => row.providerId === providerId);
}

describe("joining an existing account", () => {
  it("links Google at once for a Gmail address: Google runs that mailbox", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@gmail.com");
    answerAs("google", { email: "pat@gmail.com" });
    const browser = new Browser(handler);

    const landing = await browser.signInWith("google");

    expect(landing.pathname).toBe("/auth/callback");
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(linkedAccounts(db, "google")).toEqual([
      expect.objectContaining({
        accountId: "google-account-id",
        userId: "existing-user",
      }),
    ]);
    expect(db.UserAuthSession).toEqual([
      expect.objectContaining({ userId: "existing-user" }),
    ]);
  });

  it("links Microsoft at once for a personal account's own Outlook.com address", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@outlook.com");
    answerAs("microsoft", {
      email: "pat@outlook.com",
      microsoft: { tid: PERSONAL_TENANT },
    });

    const landing = await new Browser(handler).signInWith("microsoft");

    expect(landing.pathname).toBe("/auth/callback");
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(linkedAccounts(db, "microsoft")).toHaveLength(1);
  });

  it.each<[string, Provider, Person]>([
    [
      "a Google Workspace address",
      "google",
      { email: "pat@company.example", hd: "company.example" },
    ],
    [
      "a Google account opened with another address",
      "google",
      { email: "pat@company.example" },
    ],
    [
      "a work address its organization verified",
      "microsoft",
      {
        email: "pat@company.example",
        microsoft: { tid: "company-tenant", xms_edov: true },
      },
    ],
    [
      "a personal Microsoft account's other address",
      "microsoft",
      { email: "pat@company.example", microsoft: { tid: PERSONAL_TENANT } },
    ],
    [
      "a guest in another organization",
      "microsoft",
      {
        email: "pat@company.example",
        microsoft: { tid: "other-tenant", idp: "live.com", xms_edov: true },
      },
    ],
    ["GitHub's verified address", "github", { email: "pat@company.example" }],
    ["Discord's verified address", "discord", { email: "pat@company.example" }],
  ])("asks for a code before %s joins it", async (_case, provider, person) => {
    const { handler, db } = await createAuth();
    existingAccount(db, person.email);
    answerAs(provider, person);
    const browser = new Browser(handler);

    const landing = await browser.signInWith(provider);

    expect(landing.pathname).toBe("/login");
    expect(landing.searchParams.get("error")).toBe("sign_in_code_sent");
    expect(emailsOfType("sign_in_code")).toEqual([
      {
        to: person.email,
        type: "sign_in_code",
        code: expect.stringMatching(/^[0-9]{6}$/),
        provider,
      },
    ]);
    expect(linkedAccounts(db, provider)).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
    expect(browser.has("sign_in_code")).toBe(true);
  });

  it("links the provider account it saw once the code is entered, then signs in with one click", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@company.example");
    const person = { email: "pat@company.example", hd: "company.example" };
    answerAs("google", person);
    const browser = new Browser(handler);
    await browser.signInWith("google");

    const response = await browser.enterCode(emailedCode());

    expect(response.status).toBe(200);
    expect(await response.json()).toEqual({
      provider: "google",
      newUser: false,
    });
    expect(browser.has("session_token")).toBe(true);
    expect(browser.has("sign_in_code")).toBe(false);
    expect(linkedAccounts(db, "google")).toEqual([
      expect.objectContaining({
        accountId: "google-account-id",
        userId: "existing-user",
      }),
    ]);
    expect(db.UserAuthSession).toEqual([
      expect.objectContaining({ userId: "existing-user" }),
    ]);

    // Next time Google signs straight in.
    sendAuthEmail.mockClear();
    const next = await new Browser(handler).signInWith("google");
    expect(next.pathname).toBe("/auth/callback");
    expect(emailsOfType("sign_in_code")).toEqual([]);
  });

  it("emails the account's owner whenever a provider joins it", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@gmail.com");
    answerAs("google", { email: "pat@gmail.com" });

    await new Browser(handler).signInWith("google");

    await vi.waitFor(() =>
      expect(emailsOfType("sign_in_method_added")).toEqual([
        {
          to: "pat@gmail.com",
          type: "sign_in_method_added",
          provider: "google",
        },
      ]),
    );
  });

  it("still refuses to join an account whose own address was never verified", async () => {
    const { handler, db } = await createAuth();
    // Someone else could have signed up with the address first.
    existingAccount(db, "pat@gmail.com", false);
    answerAs("google", { email: "pat@gmail.com" });

    const landing = await new Browser(handler).signInWith("google");

    expect(landing.searchParams.get("error")).toBe("account_not_linked");
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(linkedAccounts(db, "google")).toEqual([]);
  });
});

describe("signing up", () => {
  it("creates an account at once for an address the provider verified", async () => {
    const { handler, db } = await createAuth();
    answerAs("github", { email: "pat@company.example" });

    const landing = await new Browser(handler).signInWith("github");

    expect(landing.pathname).toBe("/auth/callback");
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({ email: "pat@company.example" }),
    ]);
  });

  it("creates the account for an address the provider didn't verify once its code is entered", async () => {
    const { handler, db } = await createAuth();
    answerAs("microsoft", {
      email: "pat@gmail.example",
      microsoft: { tid: PERSONAL_TENANT },
    });
    const browser = new Browser(handler);

    const landing = await browser.signInWith("microsoft");
    expect(landing.searchParams.get("error")).toBe("sign_in_code_sent");
    expect(db.UserAuthIdentity).toEqual([]);

    const response = await browser.enterCode(emailedCode());

    expect(await response.json()).toEqual({
      provider: "microsoft",
      newUser: true,
    });
    expect(db.UserAuthIdentity).toEqual([
      expect.objectContaining({
        email: "pat@gmail.example",
        emailVerified: true,
      }),
    ]);
    expect(linkedAccounts(db, "microsoft")).toEqual([
      expect.objectContaining({ accountId: "microsoft-account-id" }),
    ]);
    expect(db.UserAuthSession).toHaveLength(1);
    // A sign-up's first sign-in method gets no notice.
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(emailsOfType("sign_in_method_added")).toEqual([]);
  });

  it("answers the same whether or not an account uses the address", async () => {
    const person = {
      email: "pat@gmail.example",
      microsoft: { tid: PERSONAL_TENANT },
    };
    const withAccount = await createAuth();
    existingAccount(withAccount.db, person.email);
    answerAs("microsoft", person);
    const first = await new Browser(withAccount.handler).signInWith(
      "microsoft",
    );
    const withoutAccount = await createAuth();
    answerAs("microsoft", person);
    const second = await new Browser(withoutAccount.handler).signInWith(
      "microsoft",
    );

    expect(first.toString()).toBe(second.toString());
  });

  it("still applies the sign-up checks when the code creates the account", async () => {
    const { handler, db } = await createAuth();
    // A team address signs up through Google only.
    answerAs("microsoft", {
      email: "pat@agpt.co",
      microsoft: { tid: PERSONAL_TENANT },
    });
    const browser = new Browser(handler);
    await browser.signInWith("microsoft");

    const response = await browser.enterCode(emailedCode());

    expect(response.status).toBe(403);
    expect(await response.json()).toMatchObject({
      code: "team_email_requires_google",
    });
    expect(db.UserAuthIdentity).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("refuses an address off the sign-up allowlist after its code, as /login expects", async () => {
    vi.stubEnv("AUTH_SIGNUP_ALLOWLIST", "invited@example.com");
    const { handler, db } = await createAuth();
    answerAs("github", { email: "pat@company.example", verified: false });
    const browser = new Browser(handler);
    await browser.signInWith("github");

    const response = await browser.enterCode(emailedCode());
    const body = await response.json();

    expect(response.status).toBe(403);
    expect(isWaitlistError(body.code, body.message)).toBe(true);
    expect(db.UserAuthIdentity).toEqual([]);
  });
});

describe("the code", () => {
  async function waitingOnCode() {
    const auth = await createAuth();
    existingAccount(auth.db, "pat@company.example");
    answerAs("github", { email: "pat@company.example" });
    const browser = new Browser(auth.handler);
    await browser.signInWith("github");
    return { ...auth, browser };
  }

  it("tells /login where it went", async () => {
    const { browser, handler } = await waitingOnCode();

    expect(await browser.pending()).toEqual({
      provider: "github",
      email: "pat@company.example",
      resendInSeconds: 60,
    });
    expect(await new Browser(handler).pending()).toBeNull();
  });

  it("refuses a wrong code, and ends the sign-in after five", async () => {
    const { browser, db } = await waitingOnCode();
    const code = emailedCode();

    for (let attempt = 1; attempt < 5; attempt++) {
      const response = await browser.enterCode(wrong(code));
      expect(response.status).toBe(400);
      expect(await response.json()).toMatchObject({
        code: "sign_in_code_invalid",
      });
    }
    const fifth = await browser.enterCode(wrong(code));
    const late = await browser.enterCode(code);

    expect(await fifth.json()).toMatchObject({ code: "sign_in_code_expired" });
    expect(await late.json()).toMatchObject({ code: "sign_in_code_expired" });
    expect(linkedAccounts(db, "github")).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("works once", async () => {
    const { browser } = await waitingOnCode();
    const code = emailedCode();

    expect((await browser.enterCode(code)).status).toBe(200);
    const again = await browser.enterCode(code);

    expect(await again.json()).toMatchObject({ code: "sign_in_code_expired" });
  });

  it("only finishes the sign-in in the browser that started it", async () => {
    const { handler, db } = await waitingOnCode();

    const elsewhere = await new Browser(handler).enterCode(emailedCode());

    expect(await elsewhere.json()).toMatchObject({
      code: "sign_in_code_expired",
    });
    expect(linkedAccounts(db, "github")).toEqual([]);
  });

  it("means nothing for another sign-in's code", async () => {
    const { handler, browser } = await waitingOnCode();
    const theirs = emailedCode();
    const other = new Browser(handler);
    await other.signInWith("github");
    const ours = emailedCode();

    const response = await browser.enterCode(
      ours === theirs ? wrong(ours) : ours,
    );

    expect(await response.json()).toMatchObject({
      code: "sign_in_code_invalid",
    });
  });

  it("expires after ten minutes", async () => {
    const { browser, db } = await waitingOnCode();
    for (const row of db.UserAuthVerification) {
      if (String(row.identifier).startsWith("sign-in-code:")) {
        row.expiresAt = new Date(Date.now() - 1000);
      }
    }

    const response = await browser.enterCode(emailedCode());

    expect(await response.json()).toMatchObject({
      code: "sign_in_code_expired",
    });
  });

  it("can be sent again after a minute, and the new one replaces it", async () => {
    const { browser, db } = await waitingOnCode();
    const first = emailedCode();

    const tooSoon = await browser.resendCode();
    expect(tooSoon.status).toBe(429);
    expect(await tooSoon.json()).toMatchObject({
      code: "sign_in_code_too_soon",
    });

    for (const row of db.UserAuthVerification) {
      if (String(row.identifier).startsWith("sign-in-code:")) {
        const record = JSON.parse(String(row.value));
        row.value = JSON.stringify({ ...record, sentAt: Date.now() - 61_000 });
      }
    }
    const resent = await browser.resendCode();
    const second = emailedCode();

    expect(resent.status).toBe(200);
    expect(emailsOfType("sign_in_code")).toHaveLength(2);
    if (second !== first) {
      expect(await (await browser.enterCode(first)).json()).toMatchObject({
        code: "sign_in_code_invalid",
      });
    }
    expect((await browser.enterCode(second)).status).toBe(200);
  });

  it("keeps the old code working when a new one can't be sent", async () => {
    const { browser, db } = await waitingOnCode();
    const first = emailedCode();
    for (const row of db.UserAuthVerification) {
      if (String(row.identifier).startsWith("sign-in-code:")) {
        const record = JSON.parse(String(row.value));
        row.value = JSON.stringify({ ...record, sentAt: Date.now() - 61_000 });
      }
    }
    sendAuthEmail.mockRejectedValueOnce(new Error("mailer down"));

    const resent = await browser.resendCode();

    expect(resent.status).toBe(503);
    expect(await resent.json()).toMatchObject({ code: "sign_in_code_unsent" });
    expect((await browser.enterCode(first)).status).toBe(200);
  });

  it("is emailed at most five times an hour to one address", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@company.example");
    answerAs("github", { email: "pat@company.example" });

    const landings = [];
    for (let attempt = 0; attempt < 6; attempt++) {
      landings.push(await new Browser(handler).signInWith("github"));
    }

    expect(
      landings.map((landing) => landing.searchParams.get("error")),
    ).toEqual([...Array(5).fill("sign_in_code_sent"), "sign_in_code_limit"]);
    expect(emailsOfType("sign_in_code")).toHaveLength(5);
  });

  it("leaves nothing behind when it can't be sent", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@company.example");
    answerAs("github", { email: "pat@company.example" });
    sendAuthEmail.mockRejectedValueOnce(new Error("mailer down"));
    const browser = new Browser(handler);

    const landing = await browser.signInWith("github");

    expect(landing.searchParams.get("error")).toBe("sign_in_code_unsent");
    expect(browser.has("sign_in_code")).toBe(false);
    expect(
      db.UserAuthVerification.filter((row) =>
        String(row.identifier).startsWith("sign-in-code:"),
      ),
    ).toEqual([]);
  });
});
