import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { Country } from "@/components/molecules/PlanCard/countries";
import { computePlanPricing } from "@/components/molecules/PlanCard/computePricing";
import {
  PLAN_KEYS,
  type PlanDef,
  type PlanKey,
} from "@/components/molecules/PlanCard/plans";
import type { useTrialCard } from "@/components/organisms/TrialCard/useTrialCard";

export interface SubscriptionPlansProps {
  plans: PlanDef[];
  country: Country;
  billing: "monthly" | "yearly";
  onBillingChange: (billing: "monthly" | "yearly") => void;
  trialOffer: TrialOfferResponse | null;
  onStartTrial: () => void;
  onSelectPlan: (key: PlanKey) => void;
  isUpdatingTier: boolean;
  selectedPlan?: string | null;
  isStartingTrial: boolean;
  trialError: string | null;
  // Names the surface for DataFast, so a shared goal never mixes clicks from
  // the onboarding paywall and the platform's upgrade modal. SubscriptionPlans
  // fills in the default before passing props down.
  goalSurface?: string;
}

export const DEFAULT_GOAL_SURFACE = "onboarding_paywall";

export type PlanDialog = "compare" | "trial" | null;

// Team is contact-sales on every surface, but onboarding's static plan list
// keys it TEAM while the API-driven paywall gets BUSINESS from the backend.
export function isTeamPlan(planKey: string) {
  return planKey === PLAN_KEYS.TEAM || planKey === PLAN_KEYS.BUSINESS;
}

/**
 * The offer to fold into the plan cards, or null to leave them priced as paid
 * plans.
 *
 * A surface may only advertise a trial it can actually deliver: the status has
 * to have loaded cleanly, the user has to still be eligible and unconverted,
 * and the offered tier has to be both trialable and present in the plans on
 * screen — otherwise the trial CTA would sit on a card the user can't reach,
 * or nowhere at all.
 */
export function getEligibleTrialOffer(
  controller: Pick<
    ReturnType<typeof useTrialCard>,
    "isLoading" | "queryError" | "trial"
  >,
  plans: PlanDef[],
) {
  const { isLoading, queryError, trial } = controller;
  const offer = trial?.offer;
  if (isLoading || queryError || !offer) return null;
  if (!trial?.eligible || trial.converted) return null;
  if (!supportsTrialPlan(offer.tier)) return null;
  return plans.some((plan) => plan.key === offer.tier) ? offer : null;
}

export function formatPlanAmount(amount: number, currency: string) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency,
    currencyDisplay: "symbol",
  }).format(amount);
}

export function supportsTrialPlan(planKey: string) {
  return planKey === PLAN_KEYS.PRO || planKey === PLAN_KEYS.MAX;
}

function getTrialPricing(offer: TrialOfferResponse) {
  const formatter = new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: offer.currency,
  });
  const decimals = formatter.resolvedOptions().maximumFractionDigits ?? 2;
  const amount = offer.unit_amount / 10 ** decimals;
  return { amount, formatted: formatter.format(amount) };
}

function getPaidActionPrice(
  trial: TrialOfferResponse | null,
  amount: number | null,
  currency: string,
  billing: SubscriptionPlansProps["billing"],
) {
  if (!trial || amount === null) return null;
  const price = formatPlanAmount(amount, currency);
  if (
    trial.currency.toUpperCase() === currency.toUpperCase() &&
    formatPlanAmount(getTrialPricing(trial).amount, currency) === price
  ) {
    return null;
  }
  return `${price} / ${billing === "yearly" ? "year" : "month"}`;
}

export function getPlanPresentation(
  plan: PlanDef,
  props: SubscriptionPlansProps,
) {
  const offer =
    supportsTrialPlan(plan.key) && props.trialOffer?.tier === plan.key
      ? props.trialOffer
      : null;
  const trial = offer?.billing_cycle === props.billing ? offer : null;
  const pricing = computePlanPricing({
    plan,
    country: props.country,
    isYearly: props.billing === "yearly",
  });
  const contactSales = pricing.primaryPrice === null;
  return {
    offer,
    trial,
    paidActionPrice: getPaidActionPrice(
      trial,
      pricing.chargedToday,
      props.country.currencyCode,
      props.billing,
    ),
    price: trial
      ? getTrialPricing(trial).formatted
      : contactSales
        ? "Contact us"
        : formatPlanAmount(
            pricing.primaryPrice ?? 0,
            props.country.currencyCode,
          ),
    unit: trial
      ? `/ ${trial.billing_cycle === "yearly" ? "year" : "month"}`
      : contactSales
        ? null
        : "/ month",
    caption: trial
      ? `after your ${trial.duration_days}-day trial · plus applicable tax`
      : contactSales
        ? null
        : props.billing === "yearly"
          ? "billed annually"
          : "billed monthly",
    action: trial ? `Start ${trial.duration_days}-day trial` : plan.cta,
  };
}
