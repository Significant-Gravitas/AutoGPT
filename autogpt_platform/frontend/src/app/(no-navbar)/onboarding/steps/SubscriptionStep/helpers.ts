import {
  PLAN_KEYS,
  PLANS,
  type PlanDef,
} from "@/components/molecules/PlanCard/plans";

export const SUBSCRIPTION_PRICING_EXPERIMENT_FLAG =
  "subscription-pricing-page-initial-state";

const HIGHLIGHT_BADGE = "Best value";

export type BillingCycle = "monthly" | "yearly";
export type HighlightedPlanKey = typeof PLAN_KEYS.PRO | typeof PLAN_KEYS.MAX;
export type SubscriptionPricingExperimentVariant =
  | "monthly-pro"
  | "monthly-max"
  | "yearly-pro"
  | "yearly-max";

interface SubscriptionPricingExperimentConfig {
  billing: BillingCycle;
  highlightedPlan: HighlightedPlanKey;
  variant: SubscriptionPricingExperimentVariant | "control";
}

const DEFAULT_EXPERIMENT_CONFIG: SubscriptionPricingExperimentConfig = {
  billing: "yearly",
  highlightedPlan: PLAN_KEYS.MAX,
  variant: "control",
};

const EXPERIMENT_CONFIGS: Record<
  SubscriptionPricingExperimentVariant,
  SubscriptionPricingExperimentConfig
> = {
  "monthly-pro": {
    billing: "monthly",
    highlightedPlan: PLAN_KEYS.PRO,
    variant: "monthly-pro",
  },
  "monthly-max": {
    billing: "monthly",
    highlightedPlan: PLAN_KEYS.MAX,
    variant: "monthly-max",
  },
  "yearly-pro": {
    billing: "yearly",
    highlightedPlan: PLAN_KEYS.PRO,
    variant: "yearly-pro",
  },
  "yearly-max": {
    billing: "yearly",
    highlightedPlan: PLAN_KEYS.MAX,
    variant: "yearly-max",
  },
};

function isExperimentVariant(
  variant: string,
): variant is SubscriptionPricingExperimentVariant {
  return variant in EXPERIMENT_CONFIGS;
}

function isPaidExperimentPlan(plan: PlanDef) {
  return plan.key === PLAN_KEYS.PRO || plan.key === PLAN_KEYS.MAX;
}

export function getSubscriptionPricingExperimentConfig(
  variant: string | boolean | undefined,
) {
  if (typeof variant === "string" && isExperimentVariant(variant)) {
    return EXPERIMENT_CONFIGS[variant];
  }
  return DEFAULT_EXPERIMENT_CONFIG;
}

export function getSubscriptionPricingExperimentPlans(
  highlightedPlan: HighlightedPlanKey,
  plans: PlanDef[] = PLANS,
) {
  return plans.map((plan) => {
    if (!isPaidExperimentPlan(plan)) return plan;

    const highlighted = plan.key === highlightedPlan;
    return {
      ...plan,
      highlighted,
      badge: highlighted ? HIGHLIGHT_BADGE : null,
      buttonVariant: highlighted ? "primary" : "secondary",
    } satisfies PlanDef;
  });
}
