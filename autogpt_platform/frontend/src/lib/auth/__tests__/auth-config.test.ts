import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("pg", () => ({ Pool: vi.fn() }));
vi.mock("better-auth", () => ({
  betterAuth: vi.fn((options: unknown) => ({ options })),
}));
vi.mock("better-auth/next-js", () => ({
  nextCookies: vi.fn(() => ({ id: "next-cookies" })),
}));
vi.mock("better-auth/plugins", () => ({
  admin: vi.fn(() => ({ id: "admin" })),
  jwt: vi.fn((opts: unknown) => ({ id: "jwt", opts })),
}));
vi.mock("../supabase-bridge", () => ({
  supabaseBridge: vi.fn(() => ({ id: "supabase-bridge" })),
}));
const sendAuthEmailMock = vi.fn();
vi.mock("../email", () => ({
  sendAuthEmail: (...args: unknown[]) => sendAuthEmailMock(...args),
}));

interface JwtPluginOptions {
  jwt: {
    audience: string;
    expirationTime: string;
    definePayload: (args: {
      user: { id: string; email: string; name: string; role?: string };
    }) => {
      email: string;
      role: string;
      user_metadata: { name: string };
    };
  };
}

interface AuthEmailArgs {
  user: { email: string };
  url: string;
}

interface CapturedAuthOptions {
  emailAndPassword: {
    minPasswordLength: number;
    revokeSessionsOnPasswordReset: boolean;
    password: {
      hash: (password: string) => Promise<string>;
      verify: (args: { hash: string; password: string }) => Promise<boolean>;
    };
    sendResetPassword: (args: AuthEmailArgs) => Promise<void>;
  };
  emailVerification: {
    sendVerificationEmail: (args: AuthEmailArgs) => Promise<void>;
  };
  advanced: { database: { generateId: () => string } };
  socialProviders: Record<string, { clientId: string; clientSecret: string }>;
  plugins: Array<{ id: string; opts?: JwtPluginOptions }>;
}

async function loadAuthOptions(): Promise<CapturedAuthOptions> {
  // The global vitest setup mocks @/lib/auth/auth to keep the pg pool out of
  // page tests; undo that here so we exercise the real config module.
  vi.doUnmock("../auth");
  vi.resetModules();
  const mod = (await import("../auth")) as unknown as {
    auth: { options: CapturedAuthOptions };
  };
  return mod.auth.options;
}

const PROVIDER_ENV_KEYS = [
  "AUTH_GOOGLE_CLIENT_ID",
  "AUTH_GOOGLE_CLIENT_SECRET",
  "AUTH_GITHUB_CLIENT_ID",
  "AUTH_GITHUB_CLIENT_SECRET",
  "AUTH_DISCORD_CLIENT_ID",
  "AUTH_DISCORD_CLIENT_SECRET",
];

beforeEach(() => {
  sendAuthEmailMock.mockReset();
  for (const key of PROVIDER_ENV_KEYS) {
    vi.stubEnv(key, "");
  }
});

afterEach(() => {
  vi.unstubAllEnvs();
});

describe("auth config", () => {
  it("revokes all sessions on password reset and keeps the GoTrue minimum password length", async () => {
    const options = await loadAuthOptions();

    expect(options.emailAndPassword.revokeSessionsOnPasswordReset).toBe(true);
    expect(options.emailAndPassword.minPasswordLength).toBe(6);
  });

  it("hashes passwords with bcrypt and verifies them round-trip", async () => {
    const options = await loadAuthOptions();
    const { hash, verify } = options.emailAndPassword.password;

    const hashed = await hash("correct horse battery staple");

    expect(hashed).toMatch(/^\$2[aby]\$10\$/);
    expect(
      await verify({ hash: hashed, password: "correct horse battery staple" }),
    ).toBe(true);
    expect(await verify({ hash: hashed, password: "wrong password" })).toBe(
      false,
    );
  });

  it("configures the JWT plugin with the Supabase-compatible audience and expiry", async () => {
    const options = await loadAuthOptions();
    const jwtPlugin = options.plugins.find((plugin) => plugin.id === "jwt");

    expect(jwtPlugin?.opts?.jwt.audience).toBe("authenticated");
    expect(jwtPlugin?.opts?.jwt.expirationTime).toBe("1h");
  });

  it("maps admin users to the admin role and everyone else to authenticated in the JWT payload", async () => {
    const options = await loadAuthOptions();
    const jwtPlugin = options.plugins.find((plugin) => plugin.id === "jwt");
    const definePayload = jwtPlugin?.opts?.jwt.definePayload;

    expect(definePayload).toBeDefined();
    expect(
      definePayload?.({
        user: {
          id: "u1",
          email: "admin@example.com",
          name: "Admin",
          role: "admin",
        },
      }),
    ).toEqual({
      email: "admin@example.com",
      role: "admin",
      user_metadata: { name: "Admin" },
    });
    expect(
      definePayload?.({
        user: {
          id: "u2",
          email: "user@example.com",
          name: "Regular",
          role: "user",
        },
      }),
    ).toEqual({
      email: "user@example.com",
      role: "authenticated",
      user_metadata: { name: "Regular" },
    });
  });

  it("generates UUID-shaped database ids", async () => {
    const options = await loadAuthOptions();

    const id = options.advanced.database.generateId();

    expect(id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i,
    );
  });

  it("registers no social providers when no provider env vars are set", async () => {
    const options = await loadAuthOptions();

    expect(options.socialProviders).toEqual({});
  });

  it("registers google when its client id and secret are configured", async () => {
    vi.stubEnv("AUTH_GOOGLE_CLIENT_ID", "google-client-id");
    vi.stubEnv("AUTH_GOOGLE_CLIENT_SECRET", "google-client-secret");

    const options = await loadAuthOptions();

    expect(options.socialProviders).toEqual({
      google: {
        clientId: "google-client-id",
        clientSecret: "google-client-secret",
      },
    });
  });

  it("sends password reset emails with the reset link", async () => {
    const options = await loadAuthOptions();

    await options.emailAndPassword.sendResetPassword({
      user: { email: "user@example.com" },
      url: "https://platform.example.com/reset?token=abc",
    });

    expect(sendAuthEmailMock).toHaveBeenCalledWith({
      to: "user@example.com",
      subject: "Reset your AutoGPT Platform password",
      text: "Click the link to reset your password: https://platform.example.com/reset?token=abc",
    });
  });

  it("sends verification emails with the verification link", async () => {
    const options = await loadAuthOptions();

    await options.emailVerification.sendVerificationEmail({
      user: { email: "new@example.com" },
      url: "https://platform.example.com/verify?token=xyz",
    });

    expect(sendAuthEmailMock).toHaveBeenCalledWith({
      to: "new@example.com",
      subject: "Verify your AutoGPT Platform email",
      text: "Click the link to verify your email: https://platform.example.com/verify?token=xyz",
    });
  });

  it("registers the supabase bridge and keeps nextCookies last in the plugin chain", async () => {
    const options = await loadAuthOptions();
    const pluginIds = options.plugins.map((plugin) => plugin.id);

    expect(pluginIds).toEqual([
      "admin",
      "jwt",
      "supabase-bridge",
      "next-cookies",
    ]);
  });
});
