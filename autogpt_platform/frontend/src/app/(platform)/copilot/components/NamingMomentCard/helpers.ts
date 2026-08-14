// No generic server-side user-preferences store exists (only notification
// preferences and the fixed OnboardingStep enum), so the "dismissed forever"
// flag lives in localStorage under a per-user key — one key per account, so
// user B dismissing on the same browser can never clobber user A's dismissal.
const NAMING_MOMENT_DISMISSED_PREFIX = "autogpt:naming-moment-dismissed";

function dismissalKey(userId: string) {
  return `${NAMING_MOMENT_DISMISSED_PREFIX}:${userId}`;
}

export function peekNamingMomentDismissed(userId: string | null | undefined) {
  if (!userId || typeof window === "undefined") return false;
  if (window.localStorage.getItem(dismissalKey(userId)) === "true") return true;
  // An earlier revision stored the last dismisser's id in one shared slot;
  // honor it so those users don't see the card again.
  return window.localStorage.getItem(NAMING_MOMENT_DISMISSED_PREFIX) === userId;
}

export function setNamingMomentDismissed(userId: string | null | undefined) {
  if (!userId || typeof window === "undefined") return;
  window.localStorage.setItem(dismissalKey(userId), "true");
}

interface EligibilityInput {
  isExpertsEnabled: boolean;
  isFlagReady: boolean;
  isLoaded: boolean;
  hasExperts: boolean;
  hasSessions: boolean;
  isDismissed: boolean;
}

// The naming moment is for an existing user whose AI has never been named:
// experts flag on, both queries settled, no expert yet (naming creates the
// first one), at least one prior chat session (proves they're not a fresh
// signup), and never dismissed.
export function isNamingMomentEligible({
  isExpertsEnabled,
  isFlagReady,
  isLoaded,
  hasExperts,
  hasSessions,
  isDismissed,
}: EligibilityInput): boolean {
  if (!isFlagReady || !isExpertsEnabled) return false;
  if (!isLoaded) return false;
  if (isDismissed) return false;
  if (hasExperts) return false;
  return hasSessions;
}
