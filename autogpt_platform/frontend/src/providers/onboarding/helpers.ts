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
