"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import {
  getGetSubscriptionStatusQueryKey,
  getGetV1ListStripeInvoicesQueryKey,
  type getSubscriptionStatusResponse,
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

import { readTier } from "../helpers";
import {
  buildPlanRequest,
  describePlanResult,
  type PlanChoiceDetails,
} from "./helpers";

export function useTrialPlanRequest(offer: TrialOfferResponse) {
  const userID = useAuthStore((state) => state.user?.id);
  const queryClient = useQueryClient();
  const { mutateAsync: updateTier } = useUpdateSubscriptionTier();
  const [requestedTier, setRequestedTier] = useState<
    PlanChoiceDetails["tier"] | null
  >(null);

  async function requestPlan(plan: PlanChoiceDetails) {
    if (!userID || requestedTier) return false;
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
      // A plan changed in another tab would turn this into a plain plan
      // change, charged without the confirmation this page shows.
      if (!(await isStillOnTrial())) {
        toast({
          title: "Your plan changed",
          description: "Review your current plan below.",
        });
        await refreshBilling();
        return false;
      }
      const response = await updateTier({
        data: buildPlanRequest(plan, offer.billing_cycle),
      });
      if (useAuthStore.getState().user?.id !== requestUserID) return false;
      const url = response.status === 200 ? response.data.url : undefined;
      if (url) {
        await leaveForCheckout(plan, url);
        return true;
      }
      const tier = response.status === 200 ? response.data.tier : undefined;
      toast(describePlanResult(plan, tier));
      await refreshBilling();
    } catch (error) {
      if (useAuthStore.getState().user?.id !== requestUserID) return false;
      // A toast, not an inline alert: the refresh below usually unmounts these
      // choices (the trial was resumed or ended elsewhere).
      toast({
        title: `Unable to start ${plan.label}`,
        description:
          error instanceof Error ? error.message : "Please try again.",
        variant: "destructive",
      });
      await refreshBilling();
    }
    return false;
  }

  async function isStillOnTrial() {
    const queryKey = getGetSubscriptionStatusQueryKey();
    await queryClient.refetchQueries({ queryKey, exact: true });
    const response =
      queryClient.getQueryData<getSubscriptionStatusResponse>(queryKey);
    return readTier(response) === "TRIAL";
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
    requestPlan,
  };
}
