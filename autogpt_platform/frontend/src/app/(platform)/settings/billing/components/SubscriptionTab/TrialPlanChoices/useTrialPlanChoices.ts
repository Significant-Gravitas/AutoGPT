"use client";

import { useEffect, useState } from "react";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { formatPlanPrice } from "@/components/organisms/TrialCard/helpers";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  trackPaywallViewed,
  trackPlanSelected,
} from "@/services/analytics/monetization-analytics";

import {
  getOwnPlanChoice,
  getUpgradePlanChoice,
  type PlanChoiceDetails,
} from "./helpers";
import { useTrialPlanRequest } from "./useTrialPlanRequest";

export function useTrialPlanChoices(offer: TrialOfferResponse) {
  const userID = useAuthStore((state) => state.user?.id);
  const subscription = useGetSubscriptionStatus({
    query: {
      select: (res) =>
        res.status === 200
          ? (res.data as SubscriptionStatusResponse)
          : undefined,
    },
  });
  const { requestedTier, error, requestPlan } = useTrialPlanRequest(offer);
  const [confirmFor, setConfirmFor] = useState<string | null>(null);

  // A trial status that lags behind a plan change must not offer the choices
  // again, so they wait for the subscription itself to still be the trial.
  const isVisible = subscription.data?.tier === "TRIAL";
  const ownPlan = getOwnPlanChoice(offer);
  const upgradePlan = subscription.data
    ? getUpgradePlanChoice(offer, subscription.data)
    : null;

  useEffect(() => {
    if (isVisible) trackPaywallViewed("billing");
  }, [isVisible]);

  function trackSelection(plan: PlanChoiceDetails) {
    trackPlanSelected({
      subscription_tier: plan.tier,
      billing_cycle: offer.billing_cycle,
      surface: "billing",
    });
  }

  async function confirmOwnPlan() {
    const isLeaving = await requestPlan(ownPlan);
    if (!isLeaving) setConfirmFor(null);
  }

  return {
    isVisible,
    ownPlan,
    upgradePlan,
    ownPrice: formatPlanPrice(offer),
    requestedTier,
    isConfirmOpen: confirmFor !== null && confirmFor === userID,
    error,
    onSelectOwnPlan: () => {
      trackSelection(ownPlan);
      setConfirmFor(userID ?? null);
    },
    onConfirmOwnPlan: () => void confirmOwnPlan(),
    onCloseConfirm: () => setConfirmFor(null),
    onSelectUpgradePlan: () => {
      if (!upgradePlan) return;
      trackSelection(upgradePlan);
      void requestPlan(upgradePlan);
    },
  };
}
