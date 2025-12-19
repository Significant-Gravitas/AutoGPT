import { getServerUser } from "@/lib/auth/server/getServerAuth";

/**
 * Get navbar account data for server-side rendering.
 *
 * Note: We intentionally do NOT prefetch the profile here because:
 * 1. Server-to-server fetch calls don't forward browser cookies automatically
 * 2. The proxy route would receive the request without auth cookies
 * 3. This would cause 401 errors
 *
 * Instead, we just check if user is logged in and let the client-side
 * React Query handle the profile fetch with proper authentication.
 */
export async function getNavbarAccountData() {
  const user = await getServerUser();
  const isLoggedIn = Boolean(user);

  return {
    isLoggedIn,
  };
}
