import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { SubscriptionTierRequest } from "@/app/api/__generated__/models/subscriptionTierRequest";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import {
  formatTrialAmount,
  trialPlanLabels,
} from "@/components/organisms/TrialCard/helpers";

import { formatCents } from "../../../helpers";
import { getCheckoutReturnURLs } from "../helpers";

type PlanTier = TrialOfferResponse["tier"];

// Team (BUSINESS) is contact-sales, so it is never offered as a self-serve
// step up from a trial.
const SELF_SERVE_TIERS: readonly PlanTier[] = ["PRO", "MAX"];

export interface PlanChoiceDetails {
  tier: PlanTier;
  label: string;
  amount: string;
  cadence: "month" | "year";
  description: string;
  cents?: number;
}

export function getCancelPendingOffer(trial: TrialStatusResponse | undefined) {
  if (!trial?.offer || !trial.active || trial.converted) return null;
  return trial.cancel_at_period_end ? trial.offer : null;
}

export function getOwnPlanChoice(offer: TrialOfferResponse): PlanChoiceDetails {
  return {
    tier: offer.tier,
    label: trialPlanLabels[offer.tier],
    amount: formatTrialAmount(offer),
    cadence: getCadence(offer),
    description: "Starts today. Your trial ends and the plan takes over.",
  };
}

export function getUpgradePlanChoice(
  offer: TrialOfferResponse,
  subscription: SubscriptionStatusResponse,
): PlanChoiceDetails | null {
  const tier = getNextSelfServeTier(offer.tier);
  if (!tier) return null;
  const costs =
    offer.billing_cycle === "yearly"
      ? subscription.tier_costs_yearly
      : subscription.tier_costs;
  const cents = costs?.[tier];
  if (cents === undefined) return null;
  return {
    tier,
    label: trialPlanLabels[tier],
    amount: formatCents(cents),
    cadence: getCadence(offer),
    description: describeUpgrade(
      offer.tier,
      tier,
      subscription.tier_multipliers,
    ),
    cents,
  };
}

export function buildPlanRequest(
  plan: PlanChoiceDetails,
  cycle: TrialOfferResponse["billing_cycle"],
): SubscriptionTierRequest {
  const { successURL, cancelURL } = getCheckoutReturnURLs({
    tier: plan.tier,
    cycle,
  });
  return {
    tier: plan.tier,
    billing_cycle: cycle,
    success_url: successURL,
    cancel_url: cancelURL,
    surface: "billing",
  };
}

function getNextSelfServeTier(tier: PlanTier) {
  const index = SELF_SERVE_TIERS.indexOf(tier);
  if (index === -1) return null;
  return SELF_SERVE_TIERS[index + 1] ?? null;
}

function describeUpgrade(
  own: PlanTier,
  upgrade: PlanTier,
  multipliers: Record<string, number> | undefined,
) {
  const ownLabel = trialPlanLabels[own];
  const ratio = (multipliers?.[upgrade] ?? 0) / (multipliers?.[own] ?? 0);
  if (!Number.isFinite(ratio) || ratio <= 1)
    return `More usage than ${ownLabel} for people who run a lot.`;
  const multiple = String(Math.round(ratio * 10) / 10);
  return `${multiple}x the usage of ${ownLabel} for people who run a lot.`;
}

function getCadence(offer: TrialOfferResponse) {
  return offer.billing_cycle === "yearly" ? "year" : "month";
}
