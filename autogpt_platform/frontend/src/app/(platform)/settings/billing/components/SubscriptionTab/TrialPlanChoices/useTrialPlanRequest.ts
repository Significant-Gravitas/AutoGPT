"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  getGetSubscriptionStatusQueryKey,
  getGetV1ListStripeInvoicesQueryKey,
  useUpdateSubscriptionTier,
} from "@/app/api/__generated__/endpoints/credits/credits";
import { getGetTrialsGetTrialStatusQueryKey } from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialOfferResponse } from "@/app/api/__generated__/models/trialOfferResponse";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useTrialFailure } from "@/components/organisms/TrialCard/useTrialFailure";
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
  const failure = useTrialFailure(userID);
  const [requestedTier, setRequestedTier] = useState<
    PlanChoiceDetails["tier"] | null
  >(null);

  async function requestPlan(plan: PlanChoiceDetails) {
    if (!userID || requestedTier) return false;
    failure.clearFailure();
    setRequestedTier(plan.tier);
    try {
      return await submitPlan(plan, userID);
    } finally {
      // Cleared after a Checkout hand-off too: a page restored from the
      // back-forward cache would otherwise keep both plans disabled.
      setRequestedTier(null);
    }
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
      await refreshBilling();
    } catch (error) {
      failure.reportFailure({
        userID: requestUserID,
        error,
        fallback: `Unable to start ${plan.label}. Please try again.`,
      });
      // A refusal usually means this page is out of date (the trial was
      // resumed or ended elsewhere), or Stripe applied a change that timed out.
      await refreshBilling();
    }
    return false;
  }

  async function refreshBilling() {
    await Promise.all(
      [
        getGetTrialsGetTrialStatusQueryKey(),
        getGetSubscriptionStatusQueryKey(),
        getGetV1ListStripeInvoicesQueryKey(),
      ].map((queryKey) => queryClient.invalidateQueries({ queryKey })),
    );
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
    error: failure.error,
    requestPlan,
  };
}
