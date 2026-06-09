import { betterAuth } from "better-auth";
import { nextCookies } from "better-auth/next-js";
import { SignJWT } from "jose";
import { PostgresDialect } from "kysely";
import { Pool } from "pg";
import { headers } from "next/headers";
import { cache } from "react";

export type LegacyAuthUser = {
  id: string;
  email?: string;
  phone?: string | null;
  role?: string;
  app_metadata?: Record<string, unknown>;
  user_metadata?: Record<string, unknown>;
  email_confirmed_at?: string | null;
  created_at?: string;
  updated_at?: string;
  name?: string | null;
  image?: string | null;
};

export type LegacyAuthSession = {
  access_token: string;
  user: LegacyAuthUser;
};

type BetterAuthSession = {
  session: {
    id: string;
    userId: string;
    expiresAt: Date;
    token: string;
    createdAt: Date;
    updatedAt: Date;
  };
  user: {
    id: string;
    email: string;
    emailVerified: boolean;
    name: string;
    image?: string | null;
    role?: string;
    createdAt: Date;
    updatedAt: Date;
  };
};

let poolSingleton: Pool | null = null;

function getDatabaseUrl() {
  return (
    process.env.DATABASE_URL ||
    process.env.DIRECT_URL ||
    "postgresql://localhost:5432/postgres"
  );
}

function getPool() {
  if (poolSingleton) {
    return poolSingleton;
  }

  poolSingleton = new Pool({
    connectionString: getDatabaseUrl(),
    max: 5,
  });

  return poolSingleton;
}

function getBaseUrl() {
  return (
    process.env.BETTER_AUTH_URL ||
    process.env.NEXT_PUBLIC_BETTER_AUTH_URL ||
    process.env.NEXT_PUBLIC_FRONTEND_BASE_URL ||
    'http://localhost:3000'
  );
}

function getTrustedOrigins() {
  const configured = process.env.BETTER_AUTH_TRUSTED_ORIGINS
    ?.split(',')
    .map((origin) => origin.trim())
    .filter(Boolean);

  return Array.from(new Set([getBaseUrl(), ...(configured || [])]));
}

function getRole(email?: string, explicitRole?: string | null) {
  return explicitRole === "admin" ? "admin" : "authenticated";
}

async function findExistingPlatformUser(email: string) {
  const result = await getPool().query<{
    id: string;
    name: string | null;
    emailVerified: boolean;
    createdAt: Date;
    updatedAt: Date;
  }>(
    'SELECT id, name, "emailVerified", "createdAt", "updatedAt" FROM platform."User" WHERE email = $1 LIMIT 1',
    [email],
  );

  return result.rows[0] ?? null;
}

function getAuthSecret() {
  return (
    process.env.BETTER_AUTH_SECRET ||
    process.env.JWT_VERIFY_KEY ||
    process.env.JWT_SECRET ||
    process.env.SUPABASE_JWT_SECRET ||
    'better-auth-secret-123456789012345678901234'
  );
}

function getProviderConfig(clientId?: string, clientSecret?: string) {
  if (!clientId || !clientSecret) {
    return undefined;
  }

  return {
    clientId,
    clientSecret,
    enabled: true,
  };
}

function getSocialProviders() {
  const providers = {
    google: getProviderConfig(
      process.env.GOOGLE_CLIENT_ID,
      process.env.GOOGLE_CLIENT_SECRET,
    ),
    github: getProviderConfig(
      process.env.GITHUB_CLIENT_ID,
      process.env.GITHUB_CLIENT_SECRET,
    ),
    discord: getProviderConfig(
      process.env.DISCORD_CLIENT_ID,
      process.env.DISCORD_CLIENT_SECRET,
    ),
  };

  return Object.fromEntries(
    Object.entries(providers).filter(([, value]) => value),
  );
}

export const auth = betterAuth({
  appName: 'AutoGPT',
  baseURL: getBaseUrl(),
  basePath: '/api/auth',
  secret: getAuthSecret(),
  database: {
    dialect: new PostgresDialect({
      pool: getPool(),
    }),
    type: 'postgres',
  },
  trustedOrigins: getTrustedOrigins(),
  emailAndPassword: {
    enabled: true,
    autoSignIn: true,
    minPasswordLength: 12,
    maxPasswordLength: 64,
    requireEmailVerification: false,
    sendResetPassword: async ({ url, user }) => {
      console.info(`Better Auth password reset for ${user.email}: ${url}`);
    },
  },
  emailVerification: {
    sendOnSignUp: false,
    autoSignInAfterVerification: true,
    sendVerificationEmail: async ({ url, user }) => {
      console.info(`Better Auth email verification for ${user.email}: ${url}`);
    },
  },
  socialProviders: getSocialProviders(),
  user: {
    additionalFields: {
      role: {
        type: 'string',
        required: false,
        input: false,
        defaultValue: 'authenticated',
      },
    },
    changeEmail: {
      enabled: true,
      updateEmailWithoutVerification: true,
    },
  },
  plugins: [nextCookies()],
  databaseHooks: {
    user: {
      create: {
        before: async (user) => {
          if (!user.email) {
            return;
          }

          const existing = await findExistingPlatformUser(user.email);
          const role = getRole(user.email, typeof user.role === 'string' ? user.role : null);

          return {
            data: {
              ...user,
              id: existing?.id ?? user.id,
              name: user.name || existing?.name || user.email.split('@')[0],
              emailVerified: existing?.emailVerified ?? Boolean(user.emailVerified),
              role,
              createdAt: existing?.createdAt ?? user.createdAt,
              updatedAt: existing?.updatedAt ?? user.updatedAt,
            },
          };
        },
      },
    },
  },
});

export function toLegacyAuthUser(session: BetterAuthSession["user"]): LegacyAuthUser {
  const role = getRole(session.email, session.role);

  return {
    id: session.id,
    email: session.email,
    role,
    name: session.name,
    image: session.image ?? null,
    app_metadata: {
      provider: 'better-auth',
      role,
    },
    user_metadata: {
      full_name: session.name,
      name: session.name,
    },
    email_confirmed_at: session.emailVerified
      ? session.updatedAt.toISOString()
      : null,
    created_at: session.createdAt.toISOString(),
    updated_at: session.updatedAt.toISOString(),
  };
}

async function signBackendJwt(session: BetterAuthSession) {
  const now = Math.floor(Date.now() / 1000);
  const role = getRole(session.user.email, session.user.role);

  return new SignJWT({
    role,
    email: session.user.email,
    aud: 'authenticated',
  })
    .setProtectedHeader({ alg: 'HS256', typ: 'JWT' })
    .setSubject(session.user.id)
    .setIssuedAt(now)
    .setExpirationTime(now + 60 * 60)
    .sign(new TextEncoder().encode(getAuthSecret()));
}

export async function toLegacyAuthSession(
  session: BetterAuthSession,
): Promise<LegacyAuthSession> {
  return {
    access_token: await signBackendJwt(session),
    user: toLegacyAuthUser(session.user),
  };
}

export const getServerAuthSession = cache(async () => {
  const requestHeaders = new Headers(await headers());
  const session = await auth.api.getSession({
    headers: requestHeaders,
    query: {
      disableCookieCache: true,
    },
  });

  return (session as BetterAuthSession | null) ?? null;
});

export async function getServerBackendToken() {
  const session = await getServerAuthSession();

  if (!session) {
    return null;
  }

  return signBackendJwt(session);
}

export function toAuthError(error: unknown) {
  const fallback = 'Authentication failed';

  if (error instanceof Error) {
    return {
      message: error.message || fallback,
      code: undefined as string | undefined,
    };
  }

  if (typeof error === 'object' && error !== null) {
    const maybeMessage = 'message' in error ? error.message : undefined;
    const maybeCode = 'code' in error ? error.code : undefined;

    return {
      message: typeof maybeMessage === 'string' ? maybeMessage : fallback,
      code: typeof maybeCode === 'string' ? maybeCode : undefined,
    };
  }

  return {
    message: fallback,
    code: undefined as string | undefined,
  };
}
