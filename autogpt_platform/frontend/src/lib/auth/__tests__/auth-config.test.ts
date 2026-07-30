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
  user: {
    changeEmail: {
      enabled: boolean;
      updateEmailWithoutVerification: boolean;
      sendChangeEmailConfirmation: (args: {
        user: { email: string };
        newEmail: string;
        url: string;
        token: string;
      }) => Promise<void>;
    };
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
  it("revokes all sessions on password reset and enforces the 12-char password floor on the raw endpoints", async () => {
    const options = await loadAuthOptions();

    expect(options.emailAndPassword.revokeSessionsOnPasswordReset).toBe(true);
    // Must be 12, not GoTrue's old 6: the Better Auth handler mounts
    // /sign-up/email and /reset-password directly, so this is the real floor
    // for every set-password path (the signup zod only guards the form).
    expect(options.emailAndPassword.minPasswordLength).toBe(12);
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
      type: "reset_password",
      url: "https://platform.example.com/reset?token=abc",
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
      type: "verify_email",
      url: "https://platform.example.com/verify?token=xyz",
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

describe("change email", () => {
  it("enables email change and routes the approval mail to the current address", async () => {
    const options = await loadAuthOptions();

    expect(options.user.changeEmail.enabled).toBe(true);

    await options.user.changeEmail.sendChangeEmailConfirmation({
      user: { email: "old@example.com" },
      newEmail: "new@example.com",
      url: "https://app/approve?token=t",
      token: "t",
    });

    expect(sendAuthEmailMock).toHaveBeenCalledWith({
      to: "old@example.com",
      type: "change_email",
      url: "https://app/approve?token=t",
    });
  });

  it("uses only change-email option names Better Auth actually supports", async () => {
    // Regression guard: the config previously used `sendChangeEmailVerification`,
    // which does not exist in Better Auth. Unknown keys are silently ignored, so
    // the approval mail quietly fell through to the default handler and went to
    // the NEW address instead of the current one — losing the anti-takeover
    // protection. Assert the supported names, and that the dead one is gone.
    const options = await loadAuthOptions();
    const changeEmail = options.user.changeEmail as unknown as Record<
      string,
      unknown
    >;

    expect(typeof changeEmail.sendChangeEmailConfirmation).toBe("function");
    expect(changeEmail.sendChangeEmailVerification).toBeUndefined();
    // Unverified users have no current address worth protecting, so their change
    // applies immediately rather than being blocked behind a mail they can't get.
    expect(changeEmail.updateEmailWithoutVerification).toBe(true);
  });
});

describe("auth table names", () => {
  it("overrides every Better Auth default table name", async () => {
    // The defaults (user, session, ...) case-collide with the platform's
    // PascalCase tables (`user` vs `User`) and read ambiguously next to them.
    // These must stay in lockstep with schema.prisma's @@map values and the
    // backend migrations — a mismatch strands Better Auth on missing tables.
    const options = (await loadAuthOptions()) as unknown as {
      user: { modelName?: string };
      session: { modelName?: string };
      account: { modelName?: string };
      verification: { modelName?: string };
      plugins: Array<{
        id: string;
        opts?: { schema?: { jwks?: { modelName?: string } } };
      }>;
    };

    expect(options.user.modelName).toBe("UserAuthIdentity");
    expect(options.session.modelName).toBe("UserAuthSession");
    expect(options.account.modelName).toBe("UserAuthAccount");
    expect(options.verification.modelName).toBe("UserAuthVerification");

    const jwtPlugin = options.plugins.find((plugin) => plugin.id === "jwt");
    expect(jwtPlugin?.opts?.schema?.jwks?.modelName).toBe("UserAuthJwks");
  });
});
