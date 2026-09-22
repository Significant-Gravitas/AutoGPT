import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { Country } from "@/components/molecules/PlanCard/countries";
import { computePlanPricing } from "@/components/molecules/PlanCard/computePricing";
import {
  PLAN_KEYS,
  type PlanDef,
  type PlanKey,
} from "@/components/molecules/PlanCard/plans";

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
}

export type PlanDialog = "compare" | "trial" | null;

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
