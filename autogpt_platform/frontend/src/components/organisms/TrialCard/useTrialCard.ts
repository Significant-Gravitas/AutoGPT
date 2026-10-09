import {
  usePostTrialsCancelTrial,
  usePostTrialsResumeTrial,
  usePostTrialsStartTrialCheckout,
} from "@/app/api/__generated__/endpoints/trials/trials";
import type { TrialStatusResponse } from "@/app/api/__generated__/models/trialStatusResponse";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { trackAdsConversionBeforeNavigation } from "@/services/analytics/google-ads";
import { markTrialCheckoutStarted } from "@/services/analytics/monetization-analytics";
import {
  type EventName,
  TrialEvent,
} from "@/services/analytics/posthog-events";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { updateTrialStatusCache } from "@/services/trials/updateTrialStatusCache";
import { useQueryClient } from "@tanstack/react-query";
import { usePostHog } from "@posthog/react";
import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { getTrialChargeAmount, getTrialDaysLeft } from "./helpers";

export function useTrialCard(returnTo: "onboarding" | "billing") {
  const userID = useAuthStore((state) => state.user?.id);
  const queryClient = useQueryClient();
  const posthog = usePostHog();
  const router = useRouter();
  const seenOffer = useRef<string | null>(null);
  const [failure, setFailure] = useState<{
    userID: string;
    message: string;
  } | null>(null);
  const query = useTrialStatus();
  const { mutateAsync: checkout, isPending: isCheckoutPending } =
    usePostTrialsStartTrialCheckout();
  // The mutation settles before the redirect, while the conversion is still
  // going out; the button must stay busy until the page actually leaves.
  const [isCheckingOut, setIsCheckingOut] = useState(false);
  const isStarting = isCheckoutPending || isCheckingOut;
  const { mutateAsync: cancel, isPending: isCanceling } =
    usePostTrialsCancelTrial();
  const { mutateAsync: resume, isPending: isResuming } =
    usePostTrialsResumeTrial();
  // Opened by this tab's own cancel, never by server state, so a reload or
  // another account never shows it.
  const [canceledFor, setCanceledFor] = useState<string | null>(null);
  const offer = query.data?.eligible ? query.data.offer : null;

  useEffect(() => {
    if (!offer || !userID) return;
    const identity = `${userID}:${offer.token}`;
    if (seenOffer.current === identity) return;
    seenOffer.current = identity;
    posthog?.capture(TrialEvent.TRIAL_OFFER_VIEWED, {
      trial_offer_version: offer.version,
      subscription_tier: offer.tier,
      trial_duration_days: offer.duration_days,
      surface: returnTo,
    });
  }, [offer, userID, posthog, returnTo]);

  async function startTrial() {
    if (!offer || !userID || isStarting) return;
    setFailure(null);
    setIsCheckingOut(true);
    try {
      const response = await checkout({
        data: { offer_token: offer.token, return_to: returnTo },
      });
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200)
        throw new Error("Unable to start trial checkout.");
      markTrialCheckoutStarted(returnTo);
      await trackAdsConversionBeforeNavigation("begin_checkout", {
        value: getTrialChargeAmount(offer),
        currency: offer.currency.toUpperCase(),
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
    } finally {
      setIsCheckingOut(false);
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
      const applied = await updateTrialStatusCache({
        queryClient,
        userID,
        response,
      });
      if (applied && isCancelPending(response.data)) {
        setCanceledFor(userID);
        posthog?.capture(TrialEvent.TRIAL_CANCEL_POPUP_VIEWED, {
          days_left: getTrialDaysLeft(response.data.ends_at),
        });
      }
    } catch (error) {
      setFailure({
        userID,
        message:
          error instanceof Error
            ? error.message
            : "Unable to cancel your trial.",
      });
      await query.refetch();
    }
  }

  async function resumeTrial() {
    if (!userID || isResuming) return;
    captureWithDaysLeft(TrialEvent.TRIAL_RESUME_CLICKED);
    setFailure(null);
    try {
      const response = await resume();
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200)
        throw new Error("Unable to resume your trial.");
      await updateTrialStatusCache({ queryClient, userID, response });
    } catch (error) {
      setFailure({
        userID,
        message:
          error instanceof Error
            ? error.message
            : "Unable to resume your trial.",
      });
      await query.refetch();
    } finally {
      setCanceledFor(null);
    }
  }

  function subscribeNow() {
    captureWithDaysLeft(TrialEvent.TRIAL_SUBSCRIBE_NOW_CLICKED);
    setCanceledFor(null);
    router.push("/settings/billing");
  }

  function dismissCanceledDialog() {
    setCanceledFor(null);
  }

  function captureWithDaysLeft(event: EventName<typeof TrialEvent>) {
    posthog?.capture(event, {
      days_left: getTrialDaysLeft(query.data?.ends_at),
    });
  }

  return {
    userID,
    trial: query.data,
    isLoading: Boolean(userID) && query.isLoading,
    error: failure && failure.userID === userID ? failure.message : null,
    queryError: query.isError,
    retry: () => query.refetch(),
    isStarting,
    isCanceling,
    isResuming,
    startTrial,
    cancelTrial,
    resumeTrial,
    showCanceledDialog: canceledFor !== null && canceledFor === userID,
    dismissCanceledDialog,
    subscribeNow,
  };
}

function isCancelPending(trial: TrialStatusResponse) {
  return Boolean(trial.active && trial.cancel_at_period_end);
}
