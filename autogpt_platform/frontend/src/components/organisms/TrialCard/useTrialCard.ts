import {
  getGetTrialsGetTrialStatusQueryKey,
  usePostTrialsCancelTrial,
  usePostTrialsStartTrialCheckout,
} from "@/app/api/__generated__/endpoints/trials/trials";
import { getGetSubscriptionStatusQueryKey } from "@/app/api/__generated__/endpoints/credits/credits";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "@posthog/react";
import { useEffect, useRef, useState } from "react";

export function useTrialCard(returnTo: "onboarding" | "billing") {
  const userID = useAuthStore((state) => state.user?.id);
  const queryClient = useQueryClient();
  const posthog = usePostHog();
  const seenOffer = useRef<string | null>(null);
  const [failure, setFailure] = useState<{
    userID: string;
    message: string;
  } | null>(null);
  const queryKey = [...getGetTrialsGetTrialStatusQueryKey(), userID];
  const query = useTrialStatus();
  const { mutateAsync: checkout, isPending: isStarting } =
    usePostTrialsStartTrialCheckout();
  const { mutateAsync: cancel, isPending: isCanceling } =
    usePostTrialsCancelTrial();
  const offer = query.data?.eligible ? query.data.offer : null;

  useEffect(() => {
    if (!offer || !userID) return;
    const identity = `${userID}:${offer.token}`;
    if (seenOffer.current === identity) return;
    seenOffer.current = identity;
    posthog?.capture("subscription_trial_offer_viewed", {
      trial_offer_version: offer.version,
      subscription_tier: offer.tier,
      trial_duration_days: offer.duration_days,
      surface: returnTo,
    });
  }, [offer, userID, posthog, returnTo]);

  async function startTrial() {
    if (!offer || !userID || isStarting) return;
    setFailure(null);
    try {
      const response = await checkout({
        data: { offer_token: offer.token, return_to: returnTo },
      });
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200)
        throw new Error("Unable to start trial checkout.");
      posthog?.capture("subscription_trial_checkout_started", {
        trial_offer_version: offer.version,
        surface: returnTo,
      });
      window.location.assign(response.data.url);
    } catch (error) {
      setFailure({
        userID,
        message:
          error instanceof Error
            ? error.message
            : "Unable to start trial checkout.",
      });
      await query.refetch();
    }
  }

  async function cancelTrial() {
    if (!userID || isCanceling) return;
    setFailure(null);
    try {
      const response = await cancel();
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200)
        throw new Error("Unable to cancel your trial.");
      queryClient.setQueryData(queryKey, response);
      await queryClient.invalidateQueries({
        queryKey: getGetSubscriptionStatusQueryKey(),
      });
    } catch (error) {
      setFailure({
        userID,
        message:
          error instanceof Error
            ? error.message
            : "Unable to cancel your trial.",
      });
    }
  }

  return {
    trial: query.data,
    isLoading: Boolean(userID) && query.isLoading,
    error: failure && failure.userID === userID ? failure.message : null,
    queryError: query.isError,
    retry: () => query.refetch(),
    isStarting,
    isCanceling,
    startTrial,
    cancelTrial,
  };
}
