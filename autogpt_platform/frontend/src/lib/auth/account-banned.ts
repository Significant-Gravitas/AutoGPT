// The backend's `ACCOUNT_BANNED_DETAIL` (autogpt_libs/auth/dependencies.py).
const ACCOUNT_BANNED_CODE = "account_banned";

const LOGOUT_REASON_PARAM = "reason";
const ACCOUNT_NOTICE_PARAM = "account";
const BANNED = "banned";

export function isAccountBannedResponse(status: number, body: unknown) {
  if (status !== 403 || typeof body !== "object" || body === null) return false;
  const detail = (body as { detail?: unknown }).detail;
  return (
    typeof detail === "object" &&
    detail !== null &&
    (detail as { code?: unknown }).code === ACCOUNT_BANNED_CODE
  );
}

// Through /logout for the full sign-out; every refused request lands here, so
// only the first one navigates.
export function signOutBannedAccount() {
  if (window.location.pathname === "/logout") return;
  window.location.replace(`/logout?${LOGOUT_REASON_PARAM}=${BANNED}`);
}

export function getPostLogoutPath(search: string) {
  const reason = new URLSearchParams(search).get(LOGOUT_REASON_PARAM);
  return reason === BANNED
    ? `/login?${ACCOUNT_NOTICE_PARAM}=${BANNED}`
    : "/login";
}

export function isBannedNotice(searchParams: URLSearchParams) {
  return searchParams.get(ACCOUNT_NOTICE_PARAM) === BANNED;
}
