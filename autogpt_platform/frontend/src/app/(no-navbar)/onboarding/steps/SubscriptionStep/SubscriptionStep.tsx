"use client";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { getEligibleTrialOffer } from "@/components/organisms/SubscriptionPlans/helpers";
import { SubscriptionPlans } from "@/components/organisms/SubscriptionPlans/SubscriptionPlans";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { useTrialCard } from "@/components/organisms/TrialCard/useTrialCard";
import { useOnboardingWizardStore } from "../../store";
import { useState } from "react";
import { useSubscriptionStep } from "./useSubscriptionStep";

export function SubscriptionStep() {
  const subscription = useSubscriptionStep();
  const trial = useTrialCard("onboarding");
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

  return (
    <FadeIn className="w-full">
      <SubscriptionPlans
        plans={subscription.plans}
        country={subscription.country}
        billing={subscription.billing}
        onBillingChange={subscription.setBilling}
        onSelectPlan={subscription.handlePlanSelect}
        isUpdatingTier={subscription.isUpdatingTier}
        selectedPlan={subscription.selectedPlan}
        trialOffer={offer}
        onStartTrial={startTrial}
        isStartingTrial={trial.isStarting || saving}
        trialError={saveError ?? trial.error}
        trialStatus={
          !offer && (
            <TrialCardContent
              returnTo="onboarding"
              controller={{
                ...trial,
                startTrial,
                isStarting: trial.isStarting || saving,
                error: saveError ?? trial.error,
              }}
            />
          )
        }
      />
    </FadeIn>
  );
}
