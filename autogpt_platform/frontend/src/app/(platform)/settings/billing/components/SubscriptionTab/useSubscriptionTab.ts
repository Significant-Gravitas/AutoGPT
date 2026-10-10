"use client";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import { useTrialStatus } from "@/services/trials/useTrialStatus";

import { isPaidTier, readTier } from "./helpers";
import { getCancelPendingOffer } from "./TrialPlanChoices/helpers";
import { usePlanCheckoutReturn } from "./usePlanCheckoutReturn";

export function useSubscriptionTab(isPlanCheckoutReturn: boolean) {
  const { data: trial, isLoading } = useTrialStatus();
  const tier = useGetSubscriptionStatus({ query: { select: readTier } });
  const { isConfirmingPlan } = usePlanCheckoutReturn(isPlanCheckoutReturn);
  // An ended trial that a paid plan replaced is history, not the plan; it
  // waits for the tier so it never flashes above the plan card.
  const isEndedTrial = Boolean(trial && !trial.eligible && !trial.active);
  const hasPaidPlan = isPaidTier(tier.data);
  const isReplaced = tier.isPending || hasPaidPlan;
  // A plan bought during a cancel-pending trial is the plan even while the
  // trial row still reads active (its stale-subscription cleanup failed).
  const isSupersededTrial = Boolean(
    trial?.active && !trial.converted && hasPaidPlan,
  );
  const cancelPendingOffer = getCancelPendingOffer(trial);

  return {
    showPlan:
      !isLoading && (!trial?.active || trial.converted || isSupersededTrial),
    showTrialCard: !(isEndedTrial && isReplaced) && !isSupersededTrial,
    planChoicesOffer: isConfirmingPlan ? null : cancelPendingOffer,
    isConfirmingPlan: Boolean(cancelPendingOffer) && isConfirmingPlan,
  };
}
