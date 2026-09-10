/**
 * Session user shape exposed to the app.
 *
 * Field names intentionally mirror the Supabase `User` object
 * (`created_at`, `user_metadata`) so the ~50 existing consumers of
 * `useAuth().user` / `getServerUser()` keep working unchanged.
 */
export interface User {
  id: string;
  email: string;
  // Snake_case to match the Supabase shape. Lets the change-email UI branch:
  // verified users get a confirmation link to their current address, unverified
  // users have the change applied immediately (see lib/auth/auth.ts changeEmail).
  email_verified?: boolean;
  role?: string;
  created_at?: string;
  user_metadata: {
    name?: string;
    email?: string;
    preferred_name?: string;
  };
}

interface SessionUserLike {
  id: string;
  email: string;
  emailVerified?: boolean | null;
  name?: string | null;
  role?: string | null;
  preferredName?: string | null;
  createdAt?: Date | string;
}

export function mapSessionUser(user: SessionUserLike): User {
  return {
    id: user.id,
    email: user.email,
    email_verified: user.emailVerified ?? undefined,
    role: user.role === "admin" ? "admin" : "authenticated",
    created_at:
      user.createdAt instanceof Date
        ? user.createdAt.toISOString()
        : user.createdAt,
    user_metadata: {
      name: user.name ?? undefined,
      email: user.email,
      preferred_name: user.preferredName ?? undefined,
    },
  };
}
