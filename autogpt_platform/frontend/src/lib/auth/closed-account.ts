// The backend's ACCOUNT_CLOSED_DETAIL (autogpt_libs/auth/closed_accounts.py).
// Every route answers it to an account its owner deleted, for the 14 days
// they can still keep it, and after.
const ACCOUNT_CLOSED_DETAIL = "This account is closed.";

// Where a closed account sees when it will be erased, with Keep my account.
export const ACCOUNT_CLOSED_PATH = "/account-closed";

export function isAccountClosedResponse(status: number, body: unknown) {
  if (status !== 401 || typeof body !== "object" || body === null) return false;
  return (body as { detail?: unknown }).detail === ACCOUNT_CLOSED_DETAIL;
}

export function isAccountClosedError(error: unknown) {
  if (typeof error !== "object" || error === null) return false;
  const { status, response } = error as {
    status?: unknown;
    response?: unknown;
  };
  return (
    typeof status === "number" && isAccountClosedResponse(status, response)
  );
}

// Every refused request lands here, so only the first one navigates. Not
// while signing out: deleting the account signs this tab out right after.
export function showAccountClosedScreen() {
  const { pathname } = window.location;
  if (pathname === ACCOUNT_CLOSED_PATH || pathname === "/logout") return;
  window.location.replace(ACCOUNT_CLOSED_PATH);
}
