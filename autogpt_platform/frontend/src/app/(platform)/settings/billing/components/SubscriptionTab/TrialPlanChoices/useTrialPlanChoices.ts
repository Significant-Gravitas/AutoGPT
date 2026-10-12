"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useState } from "react";

import { useGetSubscriptionStatus } from "@/app/api/__generated__/endpoints/credits/credits";
import type { SubscriptionStatusResponse } from "@/app/api/__generated__/models/subscriptionStatusResponse";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import {
  BILLING_PATH,
  SUBSCRIBE_NOW_PARAM,
  SUBSCRIBE_NOW_VALUE,
} from "@/components/organisms/TrialCard/helpers";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  trackPaywallViewed,
  trackPlanSelected,
} from "@/services/analytics/monetization-analytics";

import {
  findPlanChoice,
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
  const request = useTrialPlanRequest(offer);
  // Either plan charges the saved card today, so both ask first.
  const [confirming, setConfirming] = useState<{
    ownerID: string;
    tier: PlanChoiceDetails["tier"];
  } | null>(null);

  // A trial status that lags behind a plan change must not offer the choices
  // again, so they wait for the subscription itself to still be the trial.
  const isVisible = subscription.data?.tier === "TRIAL";
  const ownPlan = getOwnPlanChoice(offer);
  const upgradePlan = subscription.data
    ? getUpgradePlanChoice(offer, subscription.data)
    : null;
  const confirmPlan =
    confirming && confirming.ownerID === userID
      ? findPlanChoice([ownPlan, upgradePlan], confirming.tier)
      : null;

  useEffect(() => {
    if (isVisible) trackPaywallViewed("billing");
  }, [isVisible]);

  const router = useRouter();
  const isSubscribeNow =
    useSearchParams().get(SUBSCRIBE_NOW_PARAM) === SUBSCRIBE_NOW_VALUE;
  useEffect(() => {
    if (!isSubscribeNow || !isVisible || !userID) return;
    selectPlan(ownPlan);
    router.replace(BILLING_PATH);
  }, [isSubscribeNow, isVisible, userID]);

  function selectPlan(plan: PlanChoiceDetails) {
    trackPlanSelected({
      subscription_tier: plan.tier,
      billing_cycle: offer.billing_cycle,
      surface: "billing",
    });
    if (userID) setConfirming({ ownerID: userID, tier: plan.tier });
  }

  async function confirmPlanChoice() {
    if (confirmPlan) await request.requestPlan(confirmPlan);
    setConfirming(null);
  }

  return {
    isVisible,
    ownPlan,
    upgradePlan,
    confirmPlan,
    requestedTier: request.requestedTier,
    paymentPage: request.paymentPage,
    onSelectOwnPlan: () => selectPlan(ownPlan),
    onSelectUpgradePlan: () => {
      if (upgradePlan) selectPlan(upgradePlan);
    },
    onConfirmPlan: () => void confirmPlanChoice(),
    onCloseConfirm: () => setConfirming(null),
    onClosePaymentPage: request.closePaymentPage,
  };
}
