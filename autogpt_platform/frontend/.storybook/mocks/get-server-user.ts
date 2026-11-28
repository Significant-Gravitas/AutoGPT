/** Storybook stand-in for `@/lib/auth/server/getServerUser`: the real one reads
 *  the session through the database client. Stories render signed out. */
import type { User } from "../../src/lib/auth/types";

export async function getServerUser(): Promise<{
  user: User | null;
  role: string | null;
  error: string | null;
}> {
  return { user: null, role: null, error: null };
}
