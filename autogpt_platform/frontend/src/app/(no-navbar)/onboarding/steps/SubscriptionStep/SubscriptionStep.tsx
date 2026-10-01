"use client";

import { FadeIn } from "@/components/atoms/FadeIn/FadeIn";
import { getEligibleTrialOffer } from "@/components/organisms/SubscriptionPlans/helpers";
import { SubscriptionPlans } from "@/components/organisms/SubscriptionPlans/SubscriptionPlans";
import { TrialCardContent } from "@/components/organisms/TrialCard/TrialCard";
import { useTrialCard } from "@/components/organisms/TrialCard/useTrialCard";
import { useSubscriptionStep } from "./useSubscriptionStep";

export function SubscriptionStep() {
  const subscription = useSubscriptionStep();
  const trial = useTrialCard("onboarding");
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
        onStartTrial={trial.startTrial}
        isStartingTrial={trial.isStarting}
        trialError={trial.error}
        trialStatus={
          !offer && (
            <TrialCardContent returnTo="onboarding" controller={trial} />
          )
        }
      />
    </FadeIn>
  );
}
