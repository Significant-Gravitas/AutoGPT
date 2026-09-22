"use client";

import { useExperiment } from "@/services/experiments/useExperiment";
import { useEffect } from "react";
import { useOnboardingWizardStore } from "../../store";
import {
  getSubscriptionPricingExperimentConfig,
  getSubscriptionPricingExperimentPlans,
  SUBSCRIPTION_PRICING_EXPERIMENT_FLAG,
} from "./helpers";

export function useSubscriptionPricingExperiment() {
  const { variant, isResolved } = useExperiment(
    SUBSCRIPTION_PRICING_EXPERIMENT_FLAG,
  );
  const applyPricingExperimentBilling = useOnboardingWizardStore(
    (s) => s.applyPricingExperimentBilling,
  );
  const selectedBilling = useOnboardingWizardStore((s) => s.selectedBilling);
  const hasUserSelectedBilling = useOnboardingWizardStore(
    (s) => s.hasUserSelectedBilling,
  );
  const config = getSubscriptionPricingExperimentConfig(variant ?? undefined);

  // Wait for the arm before touching the store: applying "monthly" for an
  // unresolved flag and then flipping to the variant's cycle a moment later
  // both flashes the UI and records the wrong default.
  useEffect(() => {
    if (!isResolved) return;
    applyPricingExperimentBilling(config.billing);
  }, [isResolved, applyPricingExperimentBilling, config.billing]);

  return {
    billing: hasUserSelectedBilling ? selectedBilling : config.billing,
    plans: getSubscriptionPricingExperimentPlans(config.highlightedPlan),
    variant: config.variant,
    isResolved,
  };
}
