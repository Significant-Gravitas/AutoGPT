import { auth } from "@/lib/auth/auth";
import { cookies } from "next/headers";

/**
 * Revokes the session created earlier in the SAME server action (e.g. when
 * platform user provisioning fails right after sign-in/sign-up, the UI
 * reports failure — an authenticated cookie must not linger behind it).
 *
 * Reads cookies via `cookies()` rather than the request headers because the
 * just-set session cookie only exists in the pending response cookie store.
 * Best-effort: rollback failures are swallowed so the caller's own error
 * reporting wins.
 */
export async function rollbackSession(): Promise<void> {
  try {
    const cookieStore = await cookies();
    const cookieHeader = cookieStore
      .getAll()
      .map(({ name, value }) => `${name}=${encodeURIComponent(value)}`)
      .join("; ");

    await auth.api.signOut({
      headers: new Headers({ cookie: cookieHeader }),
    });

    for (const name of [
      "better-auth.session_token",
      "__Secure-better-auth.session_token",
      "better-auth.session_data",
      "__Secure-better-auth.session_data",
    ]) {
      if (cookieStore.get(name)) cookieStore.delete(name);
    }
  } catch (error) {
    console.error("Failed to roll back auth session:", error);
  }
}
