"use client";

import { Experiment } from "@/app/api/__generated__/models/experiment";
import { useExperiment } from "@/services/experiments/useExperiment";
import { useEffect } from "react";
import { useOnboardingWizardStore } from "../../store";
import {
  getReportedPricingVariant,
  getSubscriptionPricingExperimentConfig,
  getSubscriptionPricingExperimentPlans,
} from "./helpers";

export function useSubscriptionPricingExperiment() {
  const { variant, isResolved } = useExperiment(
    Experiment.SUBSCRIPTION_PRICING,
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
    pricingVariant: getReportedPricingVariant(variant, isResolved),
    isResolved,
  };
}
