"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  getGetSubscriptionStatusQueryKey,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import { getGetTrialsGetTrialStatusQueryKey } from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import {
  centsToUSD,
  getSubscriptionValue,
  trackAdsConversionBeforeNavigation,
} from "@/services/analytics/google-ads";

import { buildPlanRequest, type PlanChoiceDetails } from "./helpers";

export function useTrialPlanRequest(offer: TrialOfferResponse) {
  const userID = useAuthStore((state) => state.user?.id);
  const queryClient = useQueryClient();
  const { mutateAsync: updateTier } = useUpdateSubscriptionTier();
  // Stays set through the Checkout redirect so neither plan can be sent twice.
  const [requestedTier, setRequestedTier] = useState<
    PlanChoiceDetails["tier"] | null
  >(null);
  const [failure, setFailure] = useState<{
    userID: string;
    message: string;
  } | null>(null);

  async function requestPlan(plan: PlanChoiceDetails) {
    if (!userID || requestedTier) return false;
    setFailure(null);
    setRequestedTier(plan.tier);
    const isLeaving = await submitPlan(plan, userID);
    if (!isLeaving) setRequestedTier(null);
    return isLeaving;
  }

  async function submitPlan(plan: PlanChoiceDetails, requestUserID: string) {
    try {
      const response = await updateTier({
        data: buildPlanRequest(plan, offer.billing_cycle),
      });
      if (useAuthStore.getState().user?.id !== requestUserID) return false;
      const url = response.status === 200 ? response.data.url : undefined;
      if (url) {
        await leaveForCheckout(plan, url);
        return true;
      }
      toast({
        title: `You're on ${plan.label}`,
        description: "Your trial has ended and your plan starts today.",
      });
      await Promise.all([
        queryClient.invalidateQueries({
          queryKey: getGetTrialsGetTrialStatusQueryKey(),
        }),
        queryClient.invalidateQueries({
          queryKey: getGetSubscriptionStatusQueryKey(),
        }),
      ]);
    } catch (error) {
      setFailure({
        userID: requestUserID,
        message:
          error instanceof Error
            ? error.message
            : `Unable to start ${plan.label}. Please try again.`,
      });
    }
    return false;
  }

  async function leaveForCheckout(plan: PlanChoiceDetails, url: string) {
    await trackAdsConversionBeforeNavigation("begin_checkout", {
      value:
        centsToUSD(plan.cents) ??
        getSubscriptionValue(plan.tier, offer.billing_cycle),
    });
    window.location.assign(url);
  }

  return {
    requestedTier,
    error: failure && failure.userID === userID ? failure.message : null,
    requestPlan,
  };
}
