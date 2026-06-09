import { useUpdateSubscriptionTier } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionTierRequestTier } from "@/app/api/__generated__/models/subscriptionTierRequestTier";
import { toast } from "@/components/molecules/Toast/use-toast";
import { environment } from "@/services/environment";
import { useState } from "react";
import { PAYWALL_FIRST_STEPS, useOnboardingWizardStore } from "../../store";
import { COUNTRIES } from "@/components/molecules/PlanCard/countries";
import {
  PLAN_KEYS,
  type PlanKey,
  TEAM_INTAKE_FORM_URL,
} from "@/components/molecules/PlanCard/plans";
import { useSubscriptionPricingExperiment } from "./useSubscriptionPricingExperiment";

const PLAN_TO_TIER: Record<
  Exclude<PlanKey, typeof PLAN_KEYS.TEAM | typeof PLAN_KEYS.BUSINESS>,
  SubscriptionTierRequestTier
> = {
  PRO: "PRO",
  MAX: "MAX",
};

interface CheckoutResponse {
  url?: string;
}

export function useSubscriptionStep() {
  const setSelectedBilling = useOnboardingWizardStore(
    (s) => s.setSelectedBilling,
  );
  const selectedCountryCode = useOnboardingWizardStore(
    (s) => s.selectedCountryCode,
  );
  const setSelectedPlan = useOnboardingWizardStore((s) => s.setSelectedPlan);
  const nextStep = useOnboardingWizardStore((s) => s.nextStep);
  const selectedPlan = useOnboardingWizardStore((s) => s.selectedPlan);

  const { mutateAsync: updateTier, isPending: isUpdatingTier } =
    useUpdateSubscriptionTier();
  const { billing, plans } = useSubscriptionPricingExperiment();
  // Local guard that flips synchronously on first click so the profile-save
  // phase (which runs before `isUpdatingTier` becomes true) can't be
  // re-entered by a fast double-click queueing duplicate POSTs.
  const [isSubmitting, setIsSubmitting] = useState(false);
  const isProcessing = isUpdatingTier || isSubmitting;

  const countryIdx = Math.max(
    0,
    COUNTRIES.findIndex((c) => c.countryCode === selectedCountryCode),
  );
  const country = COUNTRIES[countryIdx];
  const isYearly = billing === "yearly";

  async function handlePlanSelect(planKey: PlanKey) {
    if (planKey === PLAN_KEYS.TEAM) {
      window.open(TEAM_INTAKE_FORM_URL, "_blank", "noopener,noreferrer");
      return;
    }
    if (planKey === PLAN_KEYS.BUSINESS) return;
    if (isProcessing) return;
    setIsSubmitting(true);

    // Local dev: backend has no Stripe wiring, so skip the checkout
    // round-trip and advance straight to Preparing. The profile still
    // gets POSTed there via useOnboardingPage's submission effect.
    if (environment.isLocal()) {
      setSelectedPlan(planKey);
      nextStep();
      return;
    }

    setSelectedPlan(planKey);
    const tier = PLAN_TO_TIER[planKey];

    try {
      // The paywall is the first step, so there's no profile to submit yet —
      // name / role / pain points are collected after payment. On a successful
      // checkout Stripe returns the user to Welcome to begin onboarding; on
      // cancel, back to this paywall.
      const baseUrl = `${window.location.origin}/onboarding`;
      const result = await updateTier({
        data: {
          tier,
          success_url: `${baseUrl}?step=${PAYWALL_FIRST_STEPS.welcome}&subscription=success`,
          cancel_url: `${baseUrl}?step=${PAYWALL_FIRST_STEPS.subscription}&subscription=cancelled`,
          billing_cycle: isYearly ? "yearly" : "monthly",
        },
      });
      const url = (result?.data as CheckoutResponse | undefined)?.url;
      if (url) {
        // Navigating away — don't refetch (would set state on an
        // unmounting component while Stripe Checkout takes over).
        window.location.href = url;
        return;
      }
      // Backend modified the subscription in place (no Checkout URL) —
      // proceed to the preparing step as if the user had clicked through.
      nextStep();
    } catch (error) {
      toast({
        title: "Couldn't start checkout",
        description:
          error instanceof Error
            ? error.message
            : "Stripe didn't accept the request. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsSubmitting(false);
    }
  }

  return {
    billing,
    setBilling: setSelectedBilling,
    plans,
    country,
    isYearly,
    handlePlanSelect,
    isUpdatingTier: isProcessing,
    selectedPlan,
  };
}
