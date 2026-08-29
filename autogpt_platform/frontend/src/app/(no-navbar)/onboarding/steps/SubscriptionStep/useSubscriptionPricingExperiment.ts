"use client";

import { useFeatureFlagVariantKey } from "@posthog/react";
import { useEffect } from "react";
import { useOnboardingWizardStore } from "../../store";
import {
  getSubscriptionPricingExperimentConfig,
  getSubscriptionPricingExperimentPlans,
  SUBSCRIPTION_PRICING_EXPERIMENT_FLAG,
} from "./helpers";

export function useSubscriptionPricingExperiment() {
  const variant = useFeatureFlagVariantKey(
    SUBSCRIPTION_PRICING_EXPERIMENT_FLAG,
  );
  const applyPricingExperimentBilling = useOnboardingWizardStore(
    (s) => s.applyPricingExperimentBilling,
  );
  const selectedBilling = useOnboardingWizardStore((s) => s.selectedBilling);
  const hasUserSelectedBilling = useOnboardingWizardStore(
    (s) => s.hasUserSelectedBilling,
  );
  const config = getSubscriptionPricingExperimentConfig(variant);

  useEffect(() => {
    applyPricingExperimentBilling(config.billing);
  }, [applyPricingExperimentBilling, config.billing]);

  return {
    billing: hasUserSelectedBilling ? selectedBilling : config.billing,
    plans: getSubscriptionPricingExperimentPlans(config.highlightedPlan),
    variant: config.variant,
  };
}
