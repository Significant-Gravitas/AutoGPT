import {
  peekIntroPath,
  peekWelcomePending,
} from "@/services/onboarding/brain-dump-handoff";

export const EXPERTS_ROLLOUT_AT = "2026-09-21T00:00:00Z";

const SEEN_KEY_PREFIX = "autogpt:workflows-moved:2026-09-21:";
const LANDING_ROUTES = new Set(["/copilot", "/team", "/library"]);
const CHAT_INTENT_PARAMS = [
  "sessionId",
  "expertId",
  "new",
  "kickoff",
  "autosubmit",
  "modal",
];

export function isPreExpertsUser(createdAt: string | undefined) {
  if (!createdAt) return false;
  const createdTimestamp = Date.parse(createdAt);
  return (
    Number.isFinite(createdTimestamp) &&
    createdTimestamp < Date.parse(EXPERTS_ROLLOUT_AT)
  );
}

export function isWorkflowsMovedNoticeRoute(
  pathname: string | null,
  searchParams?: Pick<URLSearchParams, "has">,
) {
  const path = pathname?.replace(/\/$/, "");
  return Boolean(
    path &&
      LANDING_ROUTES.has(path) &&
      (path !== "/copilot" ||
        !CHAT_INTENT_PARAMS.some((param) => searchParams?.has(param))),
  );
}

export function hasPendingChatHandoff() {
  if (typeof window === "undefined") return false;
  try {
    return (
      peekWelcomePending() ||
      Boolean(peekIntroPath()) ||
      Boolean(window.sessionStorage.getItem("importWorkflowPrompt")) ||
      new URLSearchParams(window.location.hash.slice(1)).has("prompt")
    );
  } catch {
    return true;
  }
}

export function peekWorkflowsMovedNoticeSeen(userID: string | null) {
  if (typeof window === "undefined" || !userID) return false;
  try {
    return window.localStorage.getItem(SEEN_KEY_PREFIX + userID) === "1";
  } catch {
    return false;
  }
}

export function setWorkflowsMovedNoticeSeen(userID: string | null) {
  if (typeof window === "undefined" || !userID) return;
  try {
    window.localStorage.setItem(SEEN_KEY_PREFIX + userID, "1");
  } catch {
    return;
  }
}
export function hasCompetingDialog() {
  return Boolean(
    document.querySelector(
      '[role="dialog"]:not([data-workflows-moved-notice]):not([hidden]):not([data-state="closed"]), ' +
        '[role="alertdialog"]:not([hidden]):not([data-state="closed"])',
    ),
  );
}
