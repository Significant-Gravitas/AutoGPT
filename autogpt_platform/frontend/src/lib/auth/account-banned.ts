// The backend's `ACCOUNT_BANNED_DETAIL` (autogpt_libs/auth/dependencies.py).
const ACCOUNT_BANNED_CODE = "account_banned";
// The backend's `ACCOUNT_CLOSED_DETAIL` (autogpt_libs/auth/closed_accounts.py),
// which a tab still open after the account was deleted gets.
const ACCOUNT_CLOSED_DETAIL = "This account is closed.";

const LOGOUT_REASON_PARAM = "reason";
const ACCOUNT_NOTICE_PARAM = "account";
const REFUSALS = ["banned", "closed"] as const;

export type AccountRefusal = (typeof REFUSALS)[number];

// The tab that deleted the account goes to /logout on its own; a request it
// still had in flight must not reload it there with the closed notice.
const OWN_DELETION_GRACE_MS = 10_000;
let ownDeletionAt = 0;

export function markAccountDeletedHere() {
  ownDeletionAt = Date.now();
}

// Why the backend refused the signed-in account itself, or null when the
// failure is about something else.
export function getAccountRefusal(
  status: number,
  body: unknown,
): AccountRefusal | null {
  if (typeof body !== "object" || body === null) return null;
  const detail = (body as { detail?: unknown }).detail;
  if (status === 401 && detail === ACCOUNT_CLOSED_DETAIL) return "closed";
  if (
    status === 403 &&
    typeof detail === "object" &&
    detail !== null &&
    (detail as { code?: unknown }).code === ACCOUNT_BANNED_CODE
  ) {
    return "banned";
  }
  return null;
}

// Through /logout for the full sign-out; every refused request lands here, so
// only the first one navigates.
export function signOutRefusedAccount(refusal: AccountRefusal) {
  if (window.location.pathname === "/logout") return;
  if (
    refusal === "closed" &&
    Date.now() - ownDeletionAt < OWN_DELETION_GRACE_MS
  ) {
    return;
  }
  window.location.replace(`/logout?${LOGOUT_REASON_PARAM}=${refusal}`);
}

function parseRefusal(value: string | null) {
  return REFUSALS.find((refusal) => refusal === value) ?? null;
}

export function getPostLogoutPath(search: string) {
  const refusal = parseRefusal(
    new URLSearchParams(search).get(LOGOUT_REASON_PARAM),
  );
  return refusal ? `/login?${ACCOUNT_NOTICE_PARAM}=${refusal}` : "/login";
}

export function getAccountNotice(searchParams: URLSearchParams) {
  return parseRefusal(searchParams.get(ACCOUNT_NOTICE_PARAM));
}
