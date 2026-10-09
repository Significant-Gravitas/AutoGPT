"use client";

import { useEffect, useState } from "react";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { formatTrialPrice } from "@/components/organisms/TrialCard/helpers";
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
  const subscription = useGetSubscriptionStatus({
    query: {
      select: (res) =>
        res.status === 200
          ? (res.data as SubscriptionStatusResponse)
          : undefined,
    },
  });
  const { requestedTier, error, requestPlan } = useTrialPlanRequest(offer);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);

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
    if (!isLeaving) setIsConfirmOpen(false);
  }

  return {
    isVisible,
    ownPlan,
    upgradePlan,
    ownPrice: formatTrialPrice(offer),
    requestedTier,
    isConfirmOpen,
    error,
    onSelectOwnPlan: () => {
      trackSelection(ownPlan);
      setIsConfirmOpen(true);
    },
    onConfirmOwnPlan: () => void confirmOwnPlan(),
    onCloseConfirm: () => setIsConfirmOpen(false),
    onSelectUpgradePlan: () => {
      if (!upgradePlan) return;
      trackSelection(upgradePlan);
      void requestPlan(upgradePlan);
    },
  };
}
