// Deleting an account bans it (backend data/account_deletion.py), and Better
// Auth's admin plugin refuses a banned account with this code.
export const CLOSED_ACCOUNT_ERROR_CODE = "BANNED_USER";

export const CLOSED_ACCOUNT_MESSAGE =
  "This account is closed. If you think this is a mistake, contact contact@agpt.co.";
