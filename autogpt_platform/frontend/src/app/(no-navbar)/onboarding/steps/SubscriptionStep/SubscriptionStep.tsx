"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { getEligibleTrialOffer } from "@/components/organisms/SubscriptionPlans/helpers";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { useTrialCard } from "@/components/organisms/TrialCard/useTrialCard";
import { PLAN_KEYS } from "@/components/molecules/PlanCard/plans";
import { computePlanPricing } from "@/components/molecules/PlanCard/computePricing";
import { formatWholeMoney } from "@/components/organisms/TrialCard/helpers";
import { useOnboardingWizardStore } from "../../store";
import { useState } from "react";
import { PaidWelcome } from "./components/PaidWelcome";
import { TrialWelcome } from "./components/TrialWelcome/TrialWelcome";
import { useReadyExperts } from "./components/useReadyExperts";
import { useSubscriptionStep } from "./useSubscriptionStep";

export function SubscriptionStep() {
  const subscription = useSubscriptionStep();
  const trial = useTrialCard("onboarding");
  const experts = useReadyExperts();
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  async function startTrial() {
    if (saving) return;
    setSaving(true);
    setSaveError(null);
    try {
      const ownerID = useOnboardingWizardStore.getState().userID;
      await useOnboardingWizardStore.getState().flushProgress?.();
      if (useOnboardingWizardStore.getState().userID !== ownerID) return;
      await trial.startTrial();
    } catch {
      setSaveError(
        "We couldn't save your progress. Please try again before continuing to checkout.",
      );
    } finally {
      setSaving(false);
    }
  }
  const offer = getEligibleTrialOffer(trial, subscription.plans);

  if (trial.isLoading)
    return (
      <div className="w-full px-4">
        <Skeleton className="mx-auto h-[34rem] w-full max-w-[60rem] rounded-[1.75rem]" />
      </div>
    );

  if (offer)
    return (
      <TrialWelcome
        offer={offer}
        experts={experts}
        isStarting={trial.isStarting || saving}
        isBlocked={trial.isBlocked || subscription.isUpdatingTier}
        error={saveError ?? trial.error}
        onStart={startTrial}
      />
    );

  const plan = subscription.plans.find((p) => p.key === PLAN_KEYS.PRO);
  const pricing = plan
    ? computePlanPricing({
        plan,
        country: subscription.country,
        isYearly: subscription.isYearly,
      })
    : null;

  return (
    <PaidWelcome
      experts={experts}
      price={formatWholeMoney(
        pricing?.chargedToday ?? 0,
        subscription.country.currencyCode,
      )}
      cadence={subscription.isYearly ? "year" : "month"}
      isProcessing={subscription.selectedPlan === PLAN_KEYS.PRO}
      isBlocked={subscription.isUpdatingTier}
      onContinue={() => subscription.handlePlanSelect(PLAN_KEYS.PRO)}
      notice={
        <TrialCardContent
          returnTo="onboarding"
          controller={{
            ...trial,
            startTrial,
            isStarting: trial.isStarting || saving,
            isBlocked: trial.isBlocked || subscription.isUpdatingTier,
            error: saveError ?? trial.error,
          }}
        />
      }
    />
  );
}
