import {
  isAccountClosedResponse,
  showAccountClosedScreen,
} from "./closed-account";

// The backend's `ACCOUNT_BANNED_DETAIL` (autogpt_libs/auth/dependencies.py).
const ACCOUNT_BANNED_CODE = "account_banned";
// The backend's `SESSION_REVOKED_DETAIL` (backend/api/session_auth.py).
const SESSION_REVOKED_CODE = "session_revoked";

const LOGOUT_REASON_PARAM = "reason";
const ACCOUNT_NOTICE_PARAM = "account";
const BANNED = "banned";

function detailCode(body: unknown) {
  if (typeof body !== "object" || body === null) return undefined;
  const detail = (body as { detail?: unknown }).detail;
  if (typeof detail !== "object" || detail === null) return undefined;
  return (detail as { code?: unknown }).code;
}

export function isAccountBannedResponse(status: number, body: unknown) {
  return status === 403 && detailCode(body) === ACCOUNT_BANNED_CODE;
}

export function isSessionRevokedResponse(status: number, body: unknown) {
  return status === 401 && detailCode(body) === SESSION_REVOKED_CODE;
}

let sessionRecheck: Promise<void> | null = null;

// A banned account leaves through /logout. An ended session is checked with
// the auth backend first, as a refocus does: the refusal may be a late one
// for an older session, and only the check knows whether the current one is
// live. The check signs out to /login?next=… or, when the backend can't
// answer, leaves the user signed in.
export function signOutIfRefused(status: number, body: unknown) {
  if (isAccountBannedResponse(status, body)) signOutBannedAccount();
  else if (isSessionRevokedResponse(status, body)) void recheckSession();
}

// A page's refused requests arrive together; they share one check.
function recheckSession() {
  sessionRecheck ??= import("./hooks/useAuthStore")
    .then(({ useAuthStore }) =>
      useAuthStore.getState().validateSession({ force: true }),
    )
    .then(
      () => undefined,
      () => undefined,
    )
    .finally(() => {
      sessionRecheck = null;
    });
  return sessionRecheck;
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

// A refused request from the browser: a banned account or an ended session is
// signed out, and a closed account is shown when it will be erased, with Keep
// my account.
export function handleRefusedAccount(status: number, body: unknown) {
  if (isAccountClosedResponse(status, body)) showAccountClosedScreen();
  else signOutIfRefused(status, body);
}
