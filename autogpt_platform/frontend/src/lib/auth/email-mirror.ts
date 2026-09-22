import type { Pool } from "pg";

// Mirror the *verified* auth email onto the platform `User` row (a separate
// table sharing this id + schema). Better Auth applies an email change only
// after the user clicks the confirmation link and fires its update hook
// post-commit, so platform User.email — which drives notifications and Stripe —
// converges on the confirmed identity instead of an unverified value the
// settings form used to write eagerly. Idempotent (the `email <> $1` guard
// skips no-op writes); self-heals any prior drift. Narrowed to `query` so it's
// trivially testable with a fake pool.
export async function mirrorVerifiedEmailToPlatformUser(
  pool: Pick<Pool, "query">,
  user: { id: string; email: string },
) {
  try {
    await pool.query(
      'UPDATE "User" SET email = $1, "updatedAt" = NOW() WHERE id = $2 AND email <> $1',
      [user.email, user.id],
    );
  } catch (error) {
    // Non-fatal: the JWT already carries the verified email and the next user
    // update retries the mirror. Never block the auth flow on this write.
    console.error(
      "Failed to mirror verified auth email to platform User",
      error,
    );
  }
}
