// @vitest-environment node
import { afterEach, describe, expect, it, vi } from "vitest";
import {
  PROFILE_IMAGE_URL_MAX_LENGTH,
  PROFILE_NAME_MAX_LENGTH,
} from "../profile-field-lengths";
import { INVALID_PROFILE_FIELD_CODE } from "../profile-field-limits";

// Runs the real Better Auth handler with this app's auth options (from
// ../auth) on an in-memory database, as email-verification-flow.test.ts does.
// The node environment keeps the session cookie, which happy-dom strips.

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
vi.mock("../email", () => ({ sendAuthEmail: vi.fn() }));

const baseURL = "http://localhost:3000";
const TOO_LONG_NAME = "n".repeat(PROFILE_NAME_MAX_LENGTH + 1);
const TOO_LONG_IMAGE = "h".repeat(PROFILE_IMAGE_URL_MAX_LENGTH + 1);
const REFUSED = { status: 400, code: INVALID_PROFILE_FIELD_CODE };

type Fields = Record<string, unknown>;

async function loadAuth() {
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
  return auth;
}

type Auth = Awaited<ReturnType<typeof loadAuth>>;

async function post(auth: Auth, path: string, body: Fields, cookie = "") {
  const response = await auth.handler(
    new Request(`${baseURL}/api/auth${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Origin: baseURL,
        Cookie: cookie,
      },
      body: JSON.stringify(body),
    }),
  );
  const answer = await response.json().catch(() => null);
  const cookies = response.headers
    .getSetCookie()
    .map((setCookie) => setCookie.split(";")[0])
    .join("; ");
  return { status: response.status, code: answer?.code, cookies };
}

function signUp(
  auth: Auth,
  fields: Fields = {},
  email = "someone@example.com",
) {
  return post(auth, "/sign-up/email", {
    email,
    password: "a-long-enough-password", // pragma: allowlist secret
    name: "someone",
    ...fields,
  });
}

function storedUser() {
  return testDB.current.UserAuthIdentity[0] as Fields;
}

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("profile fields on sign-up", () => {
  it.each([
    ["a name over the limit", { name: TOO_LONG_NAME }],
    ["a preferredName over the limit", { preferredName: TOO_LONG_NAME }],
    ["an image over the limit", { image: TOO_LONG_IMAGE }],
    ["a preferredName that is a list", { preferredName: ["someone"] }],
  ])("refuses %s", async (_, fields) => {
    const auth = await loadAuth();

    expect(await signUp(auth, fields)).toMatchObject(REFUSED);
    expect(testDB.current.UserAuthIdentity).toHaveLength(0);
  });

  it("accepts fields as long as the limits", async () => {
    const auth = await loadAuth();
    const fields = {
      name: "n".repeat(PROFILE_NAME_MAX_LENGTH),
      preferredName: "p".repeat(PROFILE_NAME_MAX_LENGTH),
      image: "h".repeat(PROFILE_IMAGE_URL_MAX_LENGTH),
    };

    expect((await signUp(auth, fields)).status).toBe(200);
    expect(storedUser()).toMatchObject(fields);
  });

  it("refuses alike whether or not the address already has an account", async () => {
    const auth = await loadAuth();
    expect((await signUp(auth)).status).toBe(200);

    const taken = await signUp(auth, { name: TOO_LONG_NAME });
    const free = await signUp(
      auth,
      { name: TOO_LONG_NAME },
      "someone-else@example.com",
    );

    expect(taken).toMatchObject(REFUSED);
    expect(free).toEqual(taken);
  });
});

describe("profile fields on update-user", () => {
  it.each([
    ["a name over the limit", { name: TOO_LONG_NAME }],
    ["a preferredName over the limit", { preferredName: TOO_LONG_NAME }],
    ["an image over the limit", { image: TOO_LONG_IMAGE }],
    ["a name that is a number", { name: 42 }],
    ["a preferredName that is an object", { preferredName: { is: "someone" } }],
  ])("refuses %s", async (_, fields) => {
    const auth = await loadAuth();
    const { cookies } = await signUp(auth);
    const before = { ...storedUser() };

    expect(await post(auth, "/update-user", fields, cookies)).toMatchObject(
      REFUSED,
    );
    expect(storedUser()).toEqual(before);
  });

  it("still updates and clears fields within the limits", async () => {
    const auth = await loadAuth();
    const { cookies } = await signUp(auth, { preferredName: "someone" });

    const updated = await post(
      auth,
      "/update-user",
      { name: "Someone Else", preferredName: null },
      cookies,
    );

    expect(updated.status).toBe(200);
    expect(storedUser()).toMatchObject({
      name: "Someone Else",
      preferredName: null,
    });
  });
});

describe("profile fields written without a checked request body", () => {
  type UserHook = (user: Fields, ctx: { path?: string } | null) => unknown;

  async function loadUserHooks() {
    const auth = await loadAuth();
    const hooks = auth.options.databaseHooks?.user;
    return {
      create: hooks?.create?.before as unknown as UserHook,
      update: hooks?.update?.before as unknown as UserHook,
    };
  }

  it("refuses a provider's profile with an oversized name", async () => {
    const { create } = await loadUserHooks();
    const profile = {
      email: "someone@example.com",
      emailVerified: true,
      name: TOO_LONG_NAME,
    };

    await expect(
      create(profile, { path: "/callback/github" }),
    ).rejects.toMatchObject({
      statusCode: 400,
      body: { code: INVALID_PROFILE_FIELD_CODE },
    });
  });

  it("refuses an admin's update with an oversized field", async () => {
    const { update } = await loadUserHooks();

    await expect(
      update({ preferredName: TOO_LONG_NAME }, { path: "/admin/update-user" }),
    ).rejects.toMatchObject({
      statusCode: 400,
      body: { code: INVALID_PROFILE_FIELD_CODE },
    });
    await expect(
      update({ name: "Someone" }, { path: "/admin/update-user" }),
    ).resolves.toBeUndefined();
  });
});
