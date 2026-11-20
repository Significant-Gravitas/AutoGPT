// @vitest-environment node
// The browser environment's Request and Response drop Cookie and Set-Cookie,
// which carry the OAuth state and the code's sign-in from one request to the
// next.
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { AuthEmailNotSent } from "../auth-email-not-sent";
import { recordWrongCode } from "../sign-in-code-lockout";

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
// Runs before each database call, so a test can hold calls at a barrier.
const dbGate = vi.hoisted(() => ({
  current: null as null | ((method: string, args: unknown[]) => Promise<void>),
}));
// What the app hands Next's `after`: its background work, such as emails.
const backgroundTasks = vi.hoisted(() => [] as Promise<unknown>[]);
vi.mock("better-auth", async (importOriginal) => {
  const actual = await importOriginal<typeof import("better-auth")>();
  const { memoryAdapter } = await import("better-auth/adapters/memory");
  // Each call waits a turn, as a real database's round trip does, so requests
  // sent at once interleave between their reads and writes.
  const interleaved =
    (
      memory: ReturnType<typeof memoryAdapter>,
    ): ReturnType<typeof memoryAdapter> =>
    (options) => {
      const adapter = memory(options);
      return new Proxy(adapter, {
        get(target, key, receiver) {
          const value = Reflect.get(target, key, receiver);
          if (typeof value !== "function" || key === "transaction")
            return value;
          return async (...args: unknown[]) => {
            await new Promise((resolve) => setTimeout(resolve, 0));
            await dbGate.current?.(String(key), args);
            return value.apply(target, args);
          };
        },
      });
    };
  return {
    ...actual,
    betterAuth: (options: Parameters<typeof actual.betterAuth>[0]) =>
      actual.betterAuth({
        ...options,
        database: interleaved(memoryAdapter(testDB.current)),
        plugins: options.plugins?.filter(
          (plugin) => plugin.id === "admin" || plugin.id === "sign-in-code",
        ),
      }),
  };
});
vi.mock("next/server", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/server")>()),
  after: vi.fn((task: Promise<unknown>) => {
    backgroundTasks.push(task);
  }),
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

// Postgres refuses a second row with the same id, which the race-safe tries
// (auth-email-cooldown.ts) rely on; the in-memory database doesn't.
class UniqueRows extends Array<Row> {
  push(...rows: Row[]) {
    for (const row of rows) {
      if (this.some((existing) => existing.id === row.id)) {
        throw new Error(`duplicate key: ${String(row.id)}`);
      }
    }
    return super.push(...rows);
  }
}

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
    UserAuthVerification: new UniqueRows() as Row[],
  };
  testDB.current = db;
  const { auth } = await import("../auth");
  return { handler: auth.handler as Handler, db, context: await auth.$context };
}

type TestDB = Awaited<ReturnType<typeof createAuth>>["db"];

afterEach(async () => {
  vi.useRealTimers();
  dbGate.current = null;
  await backgroundTasksDone();
  vi.unstubAllEnvs();
  sendAuthEmail.mockReset();
});

// Every background task a request started, finished.
async function backgroundTasksDone() {
  while (backgroundTasks.length > 0) {
    await Promise.all(backgroundTasks.splice(0));
  }
}

// Holds the database calls `matches` picks until `parties` of them have
// arrived, so they all pass whatever they checked before any of them acts.
// Correct code may never let the last one arrive, hence the time limit.
function barrierAt(
  matches: (method: string, args: unknown[]) => boolean,
  parties: number,
) {
  let arrived = 0;
  let release = () => {};
  const released = new Promise<void>((resolve) => {
    release = resolve;
  });
  return async (method: string, args: unknown[]) => {
    if (!matches(method, args)) return;
    arrived += 1;
    if (arrived >= parties) release();
    let timer: ReturnType<typeof setTimeout> | undefined;
    try {
      await Promise.race([
        released,
        new Promise((resolve) => {
          timer = setTimeout(resolve, 2000);
        }),
      ]);
    } finally {
      clearTimeout(timer);
    }
  };
}

// The database call that writes the waiting sign-in's row by its id: a
// resend replacing its code (updateMany) or a check using it up (deleteMany).
function writesPendingRow(method: string, wanted: string, args: unknown[]) {
  const { model, where } = args[0] as {
    model?: string;
    where?: Array<{ field: string }>;
  };
  return (
    method === wanted &&
    model === "verification" &&
    where?.[0]?.field === "id" &&
    where?.[1]?.field === "value"
  );
}

// As if the code was sent `ms` ago, so a new one may be asked for.
function ageCode(db: TestDB, ms: number) {
  for (const row of db.UserAuthVerification) {
    if (String(row.identifier).startsWith("sign-in-code:")) {
      const record = JSON.parse(String(row.value));
      row.value = JSON.stringify({ ...record, sentAt: record.sentAt - ms });
    }
  }
}

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

  // Another tab of the same browser, from now on with its own cookies.
  copy() {
    const tab = new Browser(this.handler);
    for (const [name, value] of this.jar) tab.jar.set(name, value);
    return tab;
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

  // From `ip` when given, as a proxy the app trusts would say.
  enterCode(code: string, ip?: string) {
    return this.request("/api/auth/sign-in-code/verify", {
      method: "POST",
      body: JSON.stringify({ code }),
      headers: ip ? { "x-forwarded-for": ip } : undefined,
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

  it.each<[string, Provider, Person]>([
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
    [
      "GitHub's unverified address",
      "github",
      { email: "pat@company.example", verified: false },
    ],
  ])(
    "still refuses %s, and emails no code: anyone can open one",
    async (_case, provider, person) => {
      const { handler, db } = await createAuth();
      existingAccount(db, person.email);
      answerAs(provider, person);

      const landing = await new Browser(handler).signInWith(provider);

      expect(landing.searchParams.get("error")).toBe("account_not_linked");
      expect(emailsOfType("sign_in_code")).toEqual([]);
      expect(linkedAccounts(db, provider)).toEqual([]);
    },
  );

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

    // Its owner hears that Google joined the account.
    await backgroundTasksDone();
    expect(emailsOfType("sign_in_method_added")).toEqual([
      {
        to: "pat@company.example",
        type: "sign_in_method_added",
        provider: "google",
      },
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

    await backgroundTasksDone();
    expect(emailsOfType("sign_in_method_added")).toEqual([
      {
        to: "pat@gmail.com",
        type: "sign_in_method_added",
        provider: "google",
      },
    ]);
  });

  it("emails the owner of an account that had no sign-in method yet", async () => {
    const { handler, db } = await createAuth();
    // Copied from Supabase with neither a password nor a provider.
    existingAccount(db, "pat@gmail.com");
    db.UserAuthAccount.length = 0;
    answerAs("google", { email: "pat@gmail.com" });

    await new Browser(handler).signInWith("google");

    await backgroundTasksDone();
    expect(linkedAccounts(db, "google")).toHaveLength(1);
    expect(emailsOfType("sign_in_method_added")).toEqual([
      {
        to: "pat@gmail.com",
        type: "sign_in_method_added",
        provider: "google",
      },
    ]);
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
    // Its own provider is no news to a new account.
    await backgroundTasksDone();
    expect(emailsOfType("sign_in_method_added")).toEqual([]);
  });

  it("still creates no account for an address the provider didn't verify, whether or not one uses it", async () => {
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

    // The same answer either way, so it doesn't say whether one does.
    expect(first.searchParams.get("error")).toBe("account_not_linked");
    expect(first.toString()).toBe(second.toString());
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(withoutAccount.db.UserAuthIdentity).toEqual([]);
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
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:00Z"));
    const { browser, handler } = await waitingOnCode();

    expect(await browser.pending()).toEqual({
      provider: "github",
      email: "pat@company.example",
      resendInSeconds: 60,
    });
    vi.setSystemTime(new Date("2026-11-17T12:30:45Z"));
    expect(await browser.pending()).toMatchObject({ resendInSeconds: 15 });
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

    expect(await fifth.json()).toMatchObject({
      code: "sign_in_code_out_of_tries",
    });
    expect(await late.json()).toMatchObject({ code: "sign_in_code_expired" });
    expect(linkedAccounts(db, "github")).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("counts checks sent at once against the same five tries", async () => {
    vi.stubEnv("TRUST_X_FORWARDED_FOR", "true");
    const { browser, db } = await waitingOnCode();
    const code = emailedCode();
    const later = browser.copy();

    // Each from its own address, so no per-IP limit spaces them out.
    const burst = await Promise.all(
      Array.from({ length: 20 }, (_, i) =>
        browser.enterCode(wrong(code), `198.51.100.${i + 1}`),
      ),
    );
    const answers = await Promise.all(
      burst.map(async (response) => (await response.json()).code),
    );

    expect(
      answers.filter((answer) => answer === "sign_in_code_invalid").length,
    ).toBeLessThanOrEqual(4);
    // The sign-in ends once; every other answer says it's over.
    expect(
      answers.filter((answer) => answer === "sign_in_code_out_of_tries"),
    ).toHaveLength(1);
    expect(
      answers.filter(
        (answer) =>
          ![
            "sign_in_code_invalid",
            "sign_in_code_out_of_tries",
            "sign_in_code_expired",
          ].includes(answer),
      ),
    ).toEqual([]);
    expect(await (await later.enterCode(code)).json()).toMatchObject({
      code: "sign_in_code_expired",
    });
    expect(linkedAccounts(db, "github")).toEqual([]);
  });

  it("finishes once when the right code is sent twice at once", async () => {
    const { browser, db } = await waitingOnCode();
    const code = emailedCode();
    const tab = browser.copy();
    // Both checks get past the code before either removes the waiting sign-in.
    dbGate.current = barrierAt(
      (method, [query]) =>
        method === "deleteMany" &&
        (query as { model?: string }).model === "verification" &&
        (query as { where?: Array<{ field: string }> }).where?.[0]?.field ===
          "id",
      2,
    );

    const answers = await Promise.all([
      browser.enterCode(code),
      tab.enterCode(code),
    ]);
    dbGate.current = null;

    expect(answers.map((answer) => answer.status).sort()).toEqual([200, 400]);
    const loser = answers.find((answer) => answer.status === 400);
    expect(await loser?.json()).toMatchObject({ code: "sign_in_code_expired" });
    expect(linkedAccounts(db, "github")).toHaveLength(1);
    expect(db.UserAuthSession).toHaveLength(1);
  });

  it("works once", async () => {
    const { browser } = await waitingOnCode();
    const code = emailedCode();

    expect((await browser.enterCode(code)).status).toBe(200);
    const again = await browser.enterCode(code);

    expect(await again.json()).toMatchObject({ code: "sign_in_code_expired" });
  });

  it("joins nothing once the account it was sent for is gone", async () => {
    const { browser, db } = await waitingOnCode();
    const code = emailedCode();
    // The account is deleted while its code is on the way.
    db.UserAuthIdentity.length = 0;
    db.UserAuthAccount.length = 0;

    const response = await browser.enterCode(code);

    expect(response.status).toBe(403);
    expect(await response.json()).toMatchObject({ code: "signup_disabled" });
    expect(db.UserAuthIdentity).toEqual([]);
    expect(db.UserAuthAccount).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
    expect(browser.has("session_token")).toBe(false);
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

  it("sends one new code when it's asked for twice at once", async () => {
    const { browser, db } = await waitingOnCode();
    ageCode(db, 61_000);
    const tab = browser.copy();
    // Both resends read the first code before either replaces it.
    dbGate.current = barrierAt(
      (method, args) => writesPendingRow(method, "updateMany", args),
      2,
    );

    const answers = await Promise.all([browser.resendCode(), tab.resendCode()]);
    dbGate.current = null;

    expect(answers.map((answer) => answer.status).sort()).toEqual([200, 429]);
    expect(emailsOfType("sign_in_code")).toHaveLength(2);
    expect((await browser.enterCode(emailedCode())).status).toBe(200);
  });

  it("doesn't let a check of a replaced code finish, or end the new one", async () => {
    const { browser, db } = await waitingOnCode();
    const first = emailedCode();
    ageCode(db, 61_000);
    let reached = () => {};
    const atUseUp = new Promise<void>((resolve) => {
      reached = resolve;
    });
    let resume = () => {};
    const resent = new Promise<void>((resolve) => {
      resume = resolve;
    });
    // The check of the first code waits, about to use it up, while a new
    // code replaces it.
    dbGate.current = async (method, args) => {
      if (!writesPendingRow(method, "deleteMany", args)) return;
      reached();
      await resent;
    };

    const checking = browser.copy().enterCode(first);
    await atUseUp;
    const resend = await browser.resendCode();
    resume();
    const answer = await checking;
    dbGate.current = null;

    expect(resend.status).toBe(200);
    expect(await answer.json()).toMatchObject({
      code: "sign_in_code_invalid",
    });
    expect(db.UserAuthSession).toEqual([]);
    expect((await browser.enterCode(emailedCode())).status).toBe(200);
  });

  it("keeps a new code working when a check of the one it replaced can't be counted", async () => {
    const { browser, db } = await waitingOnCode();
    const first = emailedCode();
    ageCode(db, 61_000);
    let reached = () => {};
    const atCount = new Promise<void>((resolve) => {
      reached = resolve;
    });
    let resume = () => {};
    const resent = new Promise<void>((resolve) => {
      resume = resolve;
    });
    // A wrong check of the first code waits at its count while a new code
    // replaces the first, then the count fails.
    dbGate.current = async (method, [query]) => {
      const { model, data } = query as {
        model?: string;
        data?: { identifier?: unknown };
      };
      if (
        method !== "create" ||
        model !== "verification" ||
        !String(data?.identifier).startsWith("sign-in-code-wrong:")
      ) {
        return;
      }
      reached();
      await resent;
      throw new Error("database unavailable");
    };

    // Both from the same browser, so an answer that ended the sign-in would
    // drop its cookie.
    const checking = browser.enterCode(wrong(first));
    await atCount;
    const resend = await browser.resendCode();
    resume();
    const answer = await checking;
    dbGate.current = null;

    expect(resend.status).toBe(200);
    expect(await answer.json()).toMatchObject({
      code: "sign_in_code_invalid",
    });
    expect(browser.has("sign_in_code")).toBe(true);
    expect((await browser.enterCode(emailedCode())).status).toBe(200);
  });

  it("doesn't renew a sign-in that ran out while its new code was made", async () => {
    const { browser, db } = await waitingOnCode();
    ageCode(db, 61_000);
    // The sign-in runs out just after the resend reads it.
    dbGate.current = async (method, args) => {
      if (!writesPendingRow(method, "updateMany", args)) return;
      for (const row of db.UserAuthVerification) {
        if (String(row.identifier).startsWith("sign-in-code:")) {
          row.expiresAt = new Date(Date.now() - 1000);
        }
      }
    };

    const resend = await browser.resendCode();
    dbGate.current = null;

    expect(await resend.json()).toMatchObject({
      code: "sign_in_code_expired",
    });
    expect(emailsOfType("sign_in_code")).toHaveLength(1);
    expect(
      db.UserAuthVerification.filter((row) =>
        String(row.identifier).startsWith("sign-in-code-email:"),
      ),
    ).toHaveLength(1);
  });

  it("gives the hour's slot back when a new code can't be stored", async () => {
    const { browser, db } = await waitingOnCode();
    ageCode(db, 61_000);
    dbGate.current = async (method, args) => {
      if (writesPendingRow(method, "updateMany", args)) {
        throw new Error("database unavailable");
      }
    };

    const resend = await browser.resendCode();
    dbGate.current = null;

    expect(resend.status).toBe(503);
    expect(await resend.json()).toMatchObject({ code: "sign_in_code_unsent" });
    expect(emailsOfType("sign_in_code")).toHaveLength(1);
    // Only the first code's slot is left.
    expect(
      db.UserAuthVerification.filter((row) =>
        String(row.identifier).startsWith("sign-in-code-email:"),
      ),
    ).toHaveLength(1);
  });

  it("gives the hour's slot back when the waiting sign-in can't be stored", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@company.example");
    answerAs("github", { email: "pat@company.example" });
    dbGate.current = async (method, [query]) => {
      const { model, data } = query as {
        model?: string;
        data?: { identifier?: unknown };
      };
      if (
        method === "create" &&
        model === "verification" &&
        String(data?.identifier).startsWith("sign-in-code:")
      ) {
        throw new Error("database unavailable");
      }
    };

    const landing = await new Browser(handler).signInWith("github");

    expect(landing.searchParams.get("error")).toBe("sign_in_code_unsent");
    expect(emailsOfType("sign_in_code")).toEqual([]);
    expect(
      db.UserAuthVerification.filter((row) =>
        /^sign-in-code(-email)?:/.test(String(row.identifier)),
      ),
    ).toEqual([]);
  });

  it("stops sending codes to an address after twenty wrong ones in thirty days", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:00Z"));
    const { browser, handler, context } = await waitingOnCode();
    const code = emailedCode();
    // Nineteen wrong codes already, from earlier sign-ins.
    for (let count = 0; count < 19; count++) {
      await recordWrongCode(context, "pat@company.example");
    }

    const twentieth = await browser.enterCode(wrong(code));
    const past = await browser.enterCode(wrong(code));
    const again = await new Browser(handler).signInWith("github");

    expect(await twentieth.json()).toMatchObject({
      code: "sign_in_code_invalid",
    });
    expect(await past.json()).toMatchObject({ code: "sign_in_code_locked" });
    expect(again.searchParams.get("error")).toBe("sign_in_code_locked");
    expect(emailsOfType("sign_in_code")).toHaveLength(1);
  });

  it("ends the sign-in when a wrong code can't be counted", async () => {
    const { browser, db } = await waitingOnCode();
    const code = emailedCode();
    // The database fails to record the address's wrong code.
    dbGate.current = async (method, [query]) => {
      const { model, data } = query as {
        model?: string;
        data?: { identifier?: unknown };
      };
      if (
        method === "create" &&
        model === "verification" &&
        String(data?.identifier).startsWith("sign-in-code-wrong:")
      ) {
        throw new Error("database unavailable");
      }
    };

    const uncounted = await browser.enterCode(wrong(code));
    dbGate.current = null;
    const late = await browser.enterCode(code);

    expect(uncounted.status).toBe(503);
    expect(await uncounted.json()).toMatchObject({
      code: "sign_in_code_unchecked",
    });
    expect(await late.json()).toMatchObject({ code: "sign_in_code_expired" });
    expect(linkedAccounts(db, "github")).toEqual([]);
    expect(db.UserAuthSession).toEqual([]);
  });

  it("checks at most ten codes a minute from one IP address", async () => {
    vi.stubEnv("TRUST_X_FORWARDED_FOR", "true");
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:30Z"));
    const { handler } = await createAuth();
    const browser = new Browser(handler);

    const statuses = [];
    for (let attempt = 0; attempt < 11; attempt++) {
      const answer = await browser.enterCode("000000", "203.0.113.7");
      statuses.push(answer.status);
    }

    expect(statuses).toEqual([...Array(10).fill(400), 429]);
  });

  it("counts each new code toward the five an hour", async () => {
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:00Z"));
    const { browser, db } = await waitingOnCode();

    const answers = [];
    for (let resend = 0; resend < 5; resend++) {
      ageCode(db, 61_000);
      const answer = await browser.resendCode();
      answers.push(answer.ok ? "sent" : (await answer.json()).code);
    }

    expect(answers).toEqual([...Array(4).fill("sent"), "sign_in_code_limit"]);
    expect(emailsOfType("sign_in_code")).toHaveLength(5);
  });

  it("is emailed at most five times an hour to one address", async () => {
    // Mid-hour, so the six sign-ins all count in the same hour.
    vi.useFakeTimers({ toFake: ["Date"] });
    vi.setSystemTime(new Date("2026-11-17T12:30:00Z"));
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
    sendAuthEmail.mockRejectedValueOnce(new AuthEmailNotSent("mailer down"));
    const browser = new Browser(handler);

    const landing = await browser.signInWith("github");

    expect(landing.searchParams.get("error")).toBe("sign_in_code_unsent");
    expect(browser.has("sign_in_code")).toBe(false);
    // Neither the waiting sign-in nor the hour's email slot: a code that
    // never arrived doesn't count toward the five.
    expect(
      db.UserAuthVerification.filter((row) =>
        /^sign-in-code(-email)?:/.test(String(row.identifier)),
      ),
    ).toEqual([]);
  });

  it("refuses the code when the hour's count can't be read", async () => {
    const { handler, db } = await createAuth();
    existingAccount(db, "pat@company.example");
    answerAs("github", { email: "pat@company.example" });
    // The database fails to record the hour's email slot.
    dbGate.current = async (method, [query]) => {
      const { model, data } = query as {
        model?: string;
        data?: { identifier?: unknown };
      };
      if (
        method === "create" &&
        model === "verification" &&
        String(data?.identifier).startsWith("sign-in-code-email:")
      ) {
        throw new Error("database unavailable");
      }
    };

    const landing = await new Browser(handler).signInWith("github");

    expect(landing.searchParams.get("error")).toBe("sign_in_code_unsent");
    expect(emailsOfType("sign_in_code")).toEqual([]);
  });
});
