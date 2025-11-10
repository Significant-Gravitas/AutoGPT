import { GraphExecutionID, OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";
import { UserOnboarding as RawUserOnboarding } from "@/app/api/__generated__/models/userOnboarding";
import { UserOnboardingUpdate } from "@/app/api/__generated__/models/userOnboardingUpdate";

export function isToday(date: Date): boolean {
  const today = new Date();
  return (
    date.getDate() === today.getDate() &&
    date.getMonth() === today.getMonth() &&
    date.getFullYear() === today.getFullYear()
  );
}

export function isYesterday(date: Date): boolean {
  const yesterday = new Date();
  yesterday.setDate(yesterday.getDate() - 1);

  return (
    date.getDate() === yesterday.getDate() &&
    date.getMonth() === yesterday.getMonth() &&
    date.getFullYear() === yesterday.getFullYear()
  );
}

export function calculateConsecutiveDays(
  lastRunAt: Date | null,
  currentConsecutiveDays: number,
): { lastRunAt: Date; consecutiveRunDays: number } {
  const now = new Date();

  if (lastRunAt === null || isYesterday(lastRunAt)) {
    return {
      lastRunAt: now,
      consecutiveRunDays: currentConsecutiveDays + 1,
    };
  }

  if (!isToday(lastRunAt)) {
    return {
      lastRunAt: now,
      consecutiveRunDays: 1,
    };
  }

  return {
    lastRunAt: now,
    consecutiveRunDays: currentConsecutiveDays,
  };
}

export function getRunMilestoneSteps(
  newRunCount: number,
  consecutiveDays: number,
): OnboardingStep[] {
  const steps: OnboardingStep[] = [];

  if (newRunCount >= 10) steps.push("RUN_AGENTS");
  if (newRunCount >= 100) steps.push("RUN_AGENTS_100");
  if (consecutiveDays >= 3) steps.push("RUN_3_DAYS");
  if (consecutiveDays >= 14) steps.push("RUN_14_DAYS");

  return steps;
}

export function processOnboardingData(
  onboarding: RawUserOnboarding,
): UserOnboarding {
  return {
    completedSteps: onboarding.completedSteps,
    walletShown: onboarding.walletShown,
    notified: onboarding.notified,
    rewardedFor: onboarding.rewardedFor,
    usageReason: onboarding.usageReason || null,
    integrations: onboarding.integrations,
    otherIntegrations: onboarding.otherIntegrations || null,
    selectedStoreListingVersionId: onboarding.selectedStoreListingVersionId || null,
    agentInput: onboarding.agentInput as {} || null,
    onboardingAgentExecutionId: onboarding.onboardingAgentExecutionId as GraphExecutionID || null,
    lastRunAt: onboarding.lastRunAt ? new Date(onboarding.lastRunAt) : null,
    consecutiveRunDays: onboarding.consecutiveRunDays,
    agentRuns: onboarding.agentRuns,
  };
}

export function shouldRedirectFromOnboarding(
  completedSteps: OnboardingStep[],
  pathname: string,
): boolean {
  return (
    completedSteps.includes("CONGRATS") &&
    !pathname.startsWith("/onboarding/reset")
  );
}


export function updateOnboardingState(
  prevState: UserOnboarding | null,
  newState: UserOnboardingUpdate,
): UserOnboarding {
  return {
    completedSteps: prevState?.completedSteps || [],
    walletShown: newState.walletShown || prevState?.walletShown || true,
    notified: newState.notified || prevState?.notified || [],
    rewardedFor: prevState?.rewardedFor || [],
    usageReason: newState.usageReason || prevState?.usageReason || null,
    integrations: newState.integrations || prevState?.integrations || [],
    otherIntegrations: newState.otherIntegrations || prevState?.otherIntegrations || null,
    selectedStoreListingVersionId: newState.selectedStoreListingVersionId || prevState?.selectedStoreListingVersionId || null,
    agentInput: newState.agentInput as Record<string, string | number> || prevState?.agentInput || null,
    onboardingAgentExecutionId: newState.onboardingAgentExecutionId as GraphExecutionID || prevState?.onboardingAgentExecutionId || null,
    agentRuns: prevState?.agentRuns || 0,
    lastRunAt: prevState?.lastRunAt || null,
    consecutiveRunDays: prevState?.consecutiveRunDays || 0,
  };
}
