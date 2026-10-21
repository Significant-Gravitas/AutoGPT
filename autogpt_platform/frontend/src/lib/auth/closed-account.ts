// Better Auth's admin plugin refuses a banned account with this code.
export const BANNED_ACCOUNT_ERROR_CODE = "BANNED_USER";

// Deleting an account bans it with this reason (backend
// data/account_deletion.py), and signing in to it is refused with this code
// instead (closed-account-plugin.ts).
export const ACCOUNT_DELETED_BAN_REASON = "account_deleted";
export const CLOSED_ACCOUNT_ERROR_CODE = "ACCOUNT_CLOSED";

export const CLOSED_ACCOUNT_MESSAGE =
  "This account is closed. If you think this is a mistake, contact contact@agpt.co.";

export const BANNED_ACCOUNT_MESSAGE =
  "This account has been banned. If you think this is a mistake, please contact support.";
