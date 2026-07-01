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
  role?: string;
  created_at?: string;
  user_metadata: {
    name?: string;
    email?: string;
  };
}

interface SessionUserLike {
  id: string;
  email: string;
  name?: string | null;
  role?: string | null;
  createdAt?: Date | string;
}

export function mapSessionUser(user: SessionUserLike): User {
  return {
    id: user.id,
    email: user.email,
    role: user.role === "admin" ? "admin" : "authenticated",
    created_at:
      user.createdAt instanceof Date
        ? user.createdAt.toISOString()
        : user.createdAt,
    user_metadata: {
      name: user.name ?? undefined,
      email: user.email,
    },
  };
}
