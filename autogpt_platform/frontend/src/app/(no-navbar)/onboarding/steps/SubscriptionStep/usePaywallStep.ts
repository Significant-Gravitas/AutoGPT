import { computePlanPricing } from "@/components/molecules/PlanCard/computePricing";
import { PLAN_KEYS } from "@/components/molecules/PlanCard/plans";
import { getEligibleTrialOffer } from "@/components/organisms/SubscriptionPlans/helpers";
import { formatWholeMoney } from "@/components/organisms/TrialCard/helpers";
import { useTrialCard } from "@/components/organisms/TrialCard/useTrialCard";
import { useState } from "react";
import { useOnboardingWizardStore } from "../../store";
import { useSubscriptionStep } from "./useSubscriptionStep";

const SAVE_FAILED =
  "We couldn't save your progress. Please try again before continuing to checkout.";

// The paying half of the plan step: the trial offer when there is one, Pro
// checkout when there is not. Progress is saved before Stripe takes over.
export function usePaywallStep() {
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
      setSaveError(SAVE_FAILED);
    } finally {
      setSaving(false);
    }
  }

  const offer = getEligibleTrialOffer(trial, subscription.plans);
  const plan = subscription.plans.find((p) => p.key === PLAN_KEYS.PRO);
  const pricing = plan
    ? computePlanPricing({
        plan,
        country: subscription.country,
        isYearly: subscription.isYearly,
      })
    : null;
  const isStarting = trial.isStarting || saving;
  const isBlocked = trial.isBlocked || subscription.isUpdatingTier;
  const error = saveError ?? trial.error;

  return {
    isLoading: trial.isLoading,
    offer,
    startTrial,
    isStarting,
    isBlocked,
    error,
    price: formatWholeMoney(
      pricing?.chargedToday ?? 0,
      subscription.country.currencyCode,
    ),
    cadence: subscription.isYearly ? ("year" as const) : ("month" as const),
    isProcessingCheckout: subscription.selectedPlan === PLAN_KEYS.PRO,
    isUpdatingTier: subscription.isUpdatingTier,
    continueToCheckout: () => subscription.handlePlanSelect(PLAN_KEYS.PRO),
    trialController: { ...trial, startTrial, isStarting, isBlocked, error },
  };
}
