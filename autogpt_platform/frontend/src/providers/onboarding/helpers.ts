import { OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";

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

  if (newRunCount === 10) steps.push("RUN_AGENTS");
  if (newRunCount === 100) steps.push("RUN_AGENTS_100");
  if (consecutiveDays === 3) steps.push("RUN_3_DAYS");
  if (consecutiveDays === 14) steps.push("RUN_14_DAYS");

  return steps;
}

export function processOnboardingData(
  onboarding: UserOnboarding,
): UserOnboarding {
  // Patch for TRIGGER_WEBHOOK - only set on backend then overwritten by frontend
  const completeWebhook =
    onboarding.rewardedFor.includes("TRIGGER_WEBHOOK") &&
    !onboarding.completedSteps.includes("TRIGGER_WEBHOOK")
      ? (["TRIGGER_WEBHOOK"] as OnboardingStep[])
      : [];

  return {
    ...onboarding,
    completedSteps: [...completeWebhook, ...onboarding.completedSteps],
    lastRunAt: onboarding.lastRunAt ? new Date(onboarding.lastRunAt) : null,
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

export function createInitialOnboardingState(
  newState: Omit<Partial<UserOnboarding>, "rewardedFor">,
): UserOnboarding {
  return {
    completedSteps: [],
    walletShown: true,
    notified: [],
    rewardedFor: [],
    usageReason: null,
    integrations: [],
    otherIntegrations: null,
    selectedStoreListingVersionId: null,
    agentInput: null,
    onboardingAgentExecutionId: null,
    agentRuns: 0,
    lastRunAt: null,
    consecutiveRunDays: 0,
    ...newState,
  };
}
