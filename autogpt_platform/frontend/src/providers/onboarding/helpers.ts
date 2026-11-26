import {
  GraphExecutionID,
  OnboardingStep,
  UserOnboarding,
} from "@/lib/autogpt-server-api";
import { UserOnboarding as RawUserOnboarding } from "@/app/api/__generated__/models/userOnboarding";

export type LocalOnboardingStateUpdate = Omit<
  Partial<UserOnboarding>,
  | "completedSteps"
  | "rewardedFor"
  | "lastRunAt"
  | "consecutiveRunDays"
  | "agentRuns"
>;

export function fromBackendUserOnboarding(
  onboarding: RawUserOnboarding,
): UserOnboarding {
  return {
    ...onboarding,
    usageReason: onboarding.usageReason || null,
    otherIntegrations: onboarding.otherIntegrations || null,
    selectedStoreListingVersionId:
      onboarding.selectedStoreListingVersionId || null,
    agentInput:
      (onboarding.agentInput as Record<string, string | number>) || null,
    onboardingAgentExecutionId:
      (onboarding.onboardingAgentExecutionId as GraphExecutionID) || null,
    lastRunAt: onboarding.lastRunAt ? new Date(onboarding.lastRunAt) : null,
  };
}

// `completedSteps` stays string[] so legacy stored values pass through, but the
// step we gate the redirect on is checked against the union: a rename that
// misses this call site must fail the build, not silently route every
// onboarded user back into the wizard.
const ONBOARDING_COMPLETE = "ONBOARDING_COMPLETE" satisfies OnboardingStep;

// Same rule as the backend's GET /onboarding/completed.
export function hasCompletedOnboarding(completedSteps: string[]): boolean {
  return completedSteps.includes(ONBOARDING_COMPLETE);
}

export interface OnboardingCompletion {
  userID: string | null;
  isCompleted: boolean;
}

// The same object when nothing changed, so consumers don't re-render.
export function withCompletion(
  prev: OnboardingCompletion | null,
  next: OnboardingCompletion,
): OnboardingCompletion {
  return prev?.userID === next.userID && prev.isCompleted === next.isCompleted
    ? prev
    : next;
}

// Pages the provider never redirects from.
const ONBOARDING_EXEMPT_PREFIXES = [
  // The public demo for signed-out prospects; a signed-in user with
  // unfinished onboarding must not be bounced out of it.
  "/tour",
  // A recovery link signs the user in and lands here. They must stay until
  // the new password is set, or the reset link becomes a one-time sign-in.
  "/reset-password",
  // A bot account-link page (/link/<token>, /link/slack/installed). Its token
  // expires in 30 minutes and the wizard never returns here.
  "/link/",
  // Signing out from the wizard must not bounce the user back into it.
  "/logout",
];

export function getOnboardingRoute(pathname: string) {
  return {
    isOnOnboardingRoute: pathname.startsWith("/onboarding"),
    // A signed-in user on these is routed onward by the provider; otherwise
    // the page shows its `isLoggedIn` loader forever.
    isOnAuthRoute: pathname === "/signup" || pathname === "/login",
    isExempt: ONBOARDING_EXEMPT_PREFIXES.some((prefix) =>
      pathname.startsWith(prefix),
    ),
  };
}

export function shouldRedirectFromOnboarding(
  completedSteps: string[],
  pathname: string,
): boolean {
  return (
    hasCompletedOnboarding(completedSteps) &&
    !pathname.startsWith("/onboarding/reset")
  );
}

/**
 * Decide where to route a logged-in user once we know their onboarding
 * completion state.
 *
 * Returns the destination path, or `null` to leave the user where they are.
 * `null` covers two distinct cases:
 *   1. A pending `?next=…` deep link — the auth page will handle it, so we
 *      must not race it.
 *   2. The user is already on the right surface (incomplete on /onboarding,
 *      completed on any non-auth/non-onboarding page).
 */
export function decideOnboardingRedirect({
  isCompleted,
  isOnOnboardingRoute,
  isOnAuthRoute,
  hasPendingAuthDeepLink,
}: {
  isCompleted: boolean;
  isOnOnboardingRoute: boolean;
  isOnAuthRoute: boolean;
  hasPendingAuthDeepLink: boolean;
}): "/onboarding" | "/home" | null {
  if (hasPendingAuthDeepLink) return null;
  if (!isCompleted && !isOnOnboardingRoute) return "/onboarding";
  if (isCompleted && (isOnOnboardingRoute || isOnAuthRoute)) return "/home";
  return null;
}

export function updateOnboardingState(
  prevState: UserOnboarding | null,
  newState: LocalOnboardingStateUpdate,
): UserOnboarding | null {
  return {
    completedSteps: prevState?.completedSteps ?? [],
    walletShown: newState.walletShown ?? prevState?.walletShown ?? false,
    notified: newState.notified ?? prevState?.notified ?? [],
    rewardedFor: prevState?.rewardedFor ?? [],
    usageReason: newState.usageReason ?? prevState?.usageReason ?? null,
    integrations: newState.integrations ?? prevState?.integrations ?? [],
    otherIntegrations:
      newState.otherIntegrations ?? prevState?.otherIntegrations ?? null,
    selectedStoreListingVersionId:
      newState.selectedStoreListingVersionId ??
      prevState?.selectedStoreListingVersionId ??
      null,
    agentInput: newState.agentInput ?? prevState?.agentInput ?? null,
    onboardingAgentExecutionId:
      newState.onboardingAgentExecutionId ??
      prevState?.onboardingAgentExecutionId ??
      null,
    lastRunAt: prevState?.lastRunAt ?? null,
    consecutiveRunDays: prevState?.consecutiveRunDays ?? 0,
    agentRuns: prevState?.agentRuns ?? 0,
  };
}
