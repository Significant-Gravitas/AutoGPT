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

import { readTier } from "../helpers";
import {
  buildPlanRequest,
  describePlanResult,
  type PlanChoiceDetails,
} from "./helpers";

export interface PaymentPage {
  ownerID: string;
  plan: PlanChoiceDetails;
  url: string;
}

export function useTrialPlanRequest(offer: TrialOfferResponse) {
  const userID = useAuthStore((state) => state.user?.id);
  const queryClient = useQueryClient();
  const { mutateAsync: updateTier } = useUpdateSubscriptionTier();
  const [requestedTier, setRequestedTier] = useState<
    PlanChoiceDetails["tier"] | null
  >(null);
  const [paymentPage, setPaymentPage] = useState<PaymentPage | null>(null);

  async function requestPlan(plan: PlanChoiceDetails) {
    if (!userID || requestedTier) return;
    setRequestedTier(plan.tier);
    try {
      await submitPlan(plan, userID);
    } finally {
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
        return;
      }
      const response = await updateTier({
        data: buildPlanRequest(plan, offer.billing_cycle),
      });
      if (useAuthStore.getState().user?.id !== requestUserID) return;
      const data = response.status === 200 ? response.data : undefined;
      // The bank wants to confirm the charge: nothing changes until the
      // customer pays on Stripe's page.
      if (data?.url) {
        setPaymentPage({ ownerID: requestUserID, plan, url: data.url });
        return;
      }
      toast(describePlanResult(plan, data?.tier));
      await refreshBilling();
    } catch (error) {
      if (useAuthStore.getState().user?.id !== requestUserID) return;
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

  return {
    requestedTier,
    requestPlan,
    paymentPage: paymentPage?.ownerID === userID ? paymentPage : null,
    closePaymentPage: () => {
      setPaymentPage(null);
      // The customer may have paid in the other tab.
      void refreshBilling();
    },
  };
}
