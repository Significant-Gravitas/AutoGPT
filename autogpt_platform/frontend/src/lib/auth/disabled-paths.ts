// Better Auth endpoints nothing in the platform calls. Mounted, each would
// still answer any signed-in browser (the /admin ones any admin), so they are
// switched off. Our own server code calls auth.api directly, which this list
// doesn't affect.
export const DISABLED_AUTH_PATHS = [
  // The admin tools use get-user, list-users, ban-user and unban-user, and the
  // load tests' accounts come from create-user. Impersonation is our own
  // header and roles are set in the database. Deleting an account needs the
  // platform's own clean-up, which deleting only the sign-in record skips.
  "/admin/has-permission",
  "/admin/impersonate-user",
  "/admin/list-user-sessions",
  "/admin/remove-user",
  "/admin/revoke-user-session",
  "/admin/revoke-user-sessions",
  "/admin/set-role",
  "/admin/set-user-password",
  "/admin/stop-impersonating",
  "/admin/update-user",
  // No account linking, no session list or revocation, no provider tokens in
  // the browser, and passwords change through the reset email.
  "/account-info",
  "/change-password",
  "/delete-user",
  "/delete-user/callback",
  "/get-access-token",
  "/link-social",
  "/list-accounts",
  "/list-sessions",
  "/refresh-token",
  "/revoke-other-sessions",
  "/revoke-session",
  "/revoke-sessions",
  "/unlink-account",
  "/update-session",
  "/verify-password",
];
