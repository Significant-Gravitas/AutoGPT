import { betterAuth } from "better-auth";
import { APIError } from "better-auth/api";
import { nextCookies } from "better-auth/next-js";
import { admin, jwt } from "better-auth/plugins";
import { compare, hash } from "bcryptjs";
import { Pool } from "pg";
import { mirrorVerifiedEmailToPlatformUser } from "./email-mirror";
import { sendAuthEmail } from "./email";
import { JWKS_ALG } from "./service-token";
import { isSignupAllowed, readSignupGateConfig } from "./signup-gate";
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

// Shared with the databaseHooks below so the verified-email mirror reuses the
// same connection/search_path instead of opening a second pool.
// Cached on globalThis in development: `next dev` re-evaluates this module on
// every hot reload, and a fresh Pool per reload leaks its idle connections
// (the old one is never ended), eventually exhausting Postgres backends.
// Production evaluates it once, so the cache is a no-op there.
const globalForAuthDb = globalThis as { __authDbPool?: Pool };

const authDbPool =
  globalForAuthDb.__authDbPool ??
  new Pool({
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
  });

if (process.env.NODE_ENV !== "production") {
  globalForAuthDb.__authDbPool = authDbPool;
}

export const auth = betterAuth({
  baseURL,
  secret: process.env.BETTER_AUTH_SECRET,
  database: authDbPool,
  telemetry: { enabled: false },
  databaseHooks: {
    user: {
      create: {
        // Env-driven signup gate (see signup-gate.ts). Fires for both
        // email/password signup AND a first OAuth sign-in, since both create
        // a user row. Existing users and the SQL data-migration bypass it.
        // The thrown message is phrased so the frontend `isWaitlistError()`
        // maps it to the existing "not allowed" modal.
        before: async (user: { email: string }) => {
          const decision = isSignupAllowed(user.email, readSignupGateConfig());
          if (!decision.allowed) {
            throw new APIError("FORBIDDEN", {
              message: decision.reason ?? "Signups are not allowed.",
            });
          }
        },
      },
      update: {
        // updateUserByEmail (fired when a change-email link is confirmed)
        // runs this hook post-commit; mirror the now-verified email onto the
        // platform User row so notifications/Stripe track the confirmed
        // identity. See email-mirror.ts for the why.
        after: async (user: { id: string; email: string }) => {
          await mirrorVerifiedEmailToPlatformUser(authDbPool, user);
        },
      },
    },
  },
  advanced: {
    database: {
      // Keep UUID ids: platform User rows reuse the auth user id, and all
      // pre-migration ids are Supabase UUIDs.
      generateId: () => crypto.randomUUID(),
    },
  },
  session: {
    modelName: "UserAuthSession",
    expiresIn: 60 * 60 * 24 * 30, // 30 days, matching GoTrue refresh longevity
    updateAge: 60 * 60 * 24,
    cookieCache: {
      enabled: true,
      maxAge: 5 * 60,
    },
  },
  // Table names are overridden away from Better Auth's defaults (user,
  // session, ...) so the auth tables read unambiguously next to the platform
  // tables — most importantly no `user` table case-colliding with `User`.
  account: {
    modelName: "UserAuthAccount",
  },
  verification: {
    modelName: "UserAuthVerification",
  },
  emailAndPassword: {
    enabled: true,
    // The 12-char policy has to hold on the raw Better Auth endpoints too
    // (/sign-up/email, /reset-password, /change-password) — the signup server
    // action's zod schema only guards the form path, so a direct POST would
    // otherwise slip a 6-char password through. minPasswordLength is checked
    // when a password is *set* (sign-up / reset / change), never on sign-in,
    // so this does NOT lock out migrated users whose old password is shorter.
    minPasswordLength: 12,
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
        type: "reset_password",
        url,
      });
    },
  },
  emailVerification: {
    sendVerificationEmail: async ({ user, url }) => {
      await sendAuthEmail({
        to: user.email,
        type: "verify_email",
        url,
      });
    },
  },
  user: {
    modelName: "UserAuthIdentity",
    additionalFields: {
      // Onboarding's "What should I call you?" answer; surfaced to
      // consumers as user_metadata.preferred_name (see mapSessionUser).
      preferredName: {
        type: "string",
        required: false,
      },
    },
    changeEmail: {
      // Off by default in Better Auth; the settings page's email form
      // depends on it. Verified users (the migration carries Supabase's
      // email_confirmed_at across as emailVerified) approve the change via a
      // confirmation link sent to their CURRENT address. That's the
      // anti-takeover protection Supabase's secure-email-change gave us, so
      // the backend mailer (Postmark) must be configured for it to work in
      // production. Unverified users (email verification is off) have the
      // change applied immediately with no email. `sendChangeEmailVerification`
      // is NOT a Better Auth key; the real ones are `sendChangeEmailConfirmation`
      // (verified path) + `updateEmailWithoutVerification` (unverified path).
      enabled: true,
      updateEmailWithoutVerification: true,
      sendChangeEmailConfirmation: async ({
        user,
        url,
      }: {
        user: { email: string };
        newEmail: string;
        url: string;
        token: string;
      }) => {
        await sendAuthEmail({
          to: user.email,
          type: "change_email",
          url,
        });
      },
    },
  },
  socialProviders: getSocialProviders(),
  plugins: [
    admin(),
    jwt({
      schema: {
        jwks: { modelName: "UserAuthJwks" },
      },
      jwks: {
        // Shared with mintServiceToken — whichever signs first creates the
        // JWKS keypair, so the two must never disagree.
        keyPairConfig: { alg: JWKS_ALG },
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
