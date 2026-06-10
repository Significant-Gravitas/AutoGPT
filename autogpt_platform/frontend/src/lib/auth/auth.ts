import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { admin, jwt } from "better-auth/plugins";
import { compare, hash } from "bcryptjs";
import { Pool } from "pg";
import { sendAuthEmail } from "./email";
import { supabaseBridge } from "./supabase-bridge";

const baseURL =
  process.env.BETTER_AUTH_URL ||
  process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
  "http://localhost:3000";

function getSocialProviders() {
  const providers: Record<string, { clientId: string; clientSecret: string }> =
    {};

  for (const provider of ["google", "github", "discord"] as const) {
    const prefix = `AUTH_${provider.toUpperCase()}`;
    const clientId = process.env[`${prefix}_CLIENT_ID`];
    const clientSecret = process.env[`${prefix}_CLIENT_SECRET`];
    if (clientId && clientSecret) {
      providers[provider] = { clientId, clientSecret };
    }
  }

  return providers;
}

export const auth = betterAuth({
  baseURL,
  secret: process.env.BETTER_AUTH_SECRET,
  database: new Pool({
    // Fallback matches the docker-compose db service so `make run-frontend`
    // works against `make start-core` without a frontend/.env, mirroring the
    // localhost fallbacks in services/environment. Production must set
    // DATABASE_URL explicitly.
    connectionString:
      process.env.DATABASE_URL ||
      "postgresql://postgres:your-super-secret-and-long-postgres-password@localhost:5432/postgres",
    // Better Auth shares the platform Postgres; its tables live in the same
    // schema as the Prisma-managed ones (created by the backend migrations).
    options: `-c search_path=${process.env.AUTH_DB_SCHEMA || "platform"}`,
  }),
  telemetry: { enabled: false },
  advanced: {
    database: {
      // Keep UUID ids: platform User rows reuse the auth user id, and all
      // pre-migration ids are Supabase UUIDs.
      generateId: () => crypto.randomUUID(),
    },
  },
  session: {
    expiresIn: 60 * 60 * 24 * 30, // 30 days, matching GoTrue refresh longevity
    updateAge: 60 * 60 * 24,
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60,
    },
  },
  emailAndPassword: {
    enabled: true,
    // GoTrue's minimum was 6; raising this would lock out existing users at
    // sign-in. The signup form enforces 12+ client- and server-side.
    minPasswordLength: 6,
    // A password reset kicks every active session, matching the previous
    // flow's signOut({ scope: "global" }) — the standard defense when a
    // user resets their password to evict a stolen session.
    revokeSessionsOnPasswordReset: true,
    requireEmailVerification:
      process.env.AUTH_REQUIRE_EMAIL_VERIFICATION === "true",
    password: {
      // bcrypt instead of Better Auth's default scrypt so password hashes
      // migrated from Supabase GoTrue keep verifying without a reset.
      hash: (password) => hash(password, 10),
      verify: ({ hash: hashValue, password }) => compare(password, hashValue),
    },
    sendResetPassword: async ({ user, url }) => {
      await sendAuthEmail({
        to: user.email,
        subject: "Reset your AutoGPT Platform password",
        text: `Click the link to reset your password: ${url}`,
      });
    },
  },
  emailVerification: {
    sendVerificationEmail: async ({ user, url }) => {
      await sendAuthEmail({
        to: user.email,
        subject: "Verify your AutoGPT Platform email",
        text: `Click the link to verify your email: ${url}`,
      });
    },
  },
  user: {
    changeEmail: {
      // Off by default in Better Auth; the settings page's email form
      // depends on it. Verified users approve the change via a link sent
      // to their CURRENT address (anti-takeover), so SMTP must be
      // configured for email changes to work in production.
      enabled: true,
      sendChangeEmailVerification: async ({
        user,
        newEmail,
        url,
      }: {
        user: { email: string };
        newEmail: string;
        url: string;
      }) => {
        await sendAuthEmail({
          to: user.email,
          subject: "Confirm your AutoGPT Platform email change",
          text: `Click the link to approve changing your email to ${newEmail}: ${url}`,
        });
      },
    },
  },
  socialProviders: getSocialProviders(),
  plugins: [
    admin(),
    jwt({
      jwks: {
        keyPairConfig: { alg: "ES256" },
      },
      jwt: {
        issuer: baseURL,
        // The Python backend validates `aud == "authenticated"` — the same
        // audience Supabase GoTrue used, so old and new tokens are
        // interchangeable during the migration window.
        audience: "authenticated",
        expirationTime: "1h",
        definePayload: ({ user }) => ({
          email: user.email,
          role: user.role === "admin" ? "admin" : "authenticated",
          user_metadata: { name: user.name },
        }),
      },
    }),
    supabaseBridge(),
    // Must be last so cookies set inside server actions stick.
    nextCookies(),
  ],
});
