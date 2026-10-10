import { usePostTrialsStartTrialCheckout } from "@/app/api/__generated__/endpoints/trials/trials";
import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import { trackAdsConversionBeforeNavigation } from "@/services/analytics/google-ads";
import { markTrialCheckoutStarted } from "@/services/analytics/monetization-analytics";
import { useTrialStatus } from "@/services/trials/useTrialStatus";
import { useState } from "react";
import { getTrialChargeAmount } from "./helpers";
import { useTrialCancellation } from "./useTrialCancellation";
import { useTrialFailure } from "./useTrialFailure";
import { useTrialOfferViewed } from "./useTrialOfferViewed";

const CHECKOUT_FAILED = "Unable to start trial checkout.";

export function useTrialCard(returnTo: "onboarding" | "billing") {
  const userID = useAuthStore((state) => state.user?.id);
  const query = useTrialStatus();
  const failure = useTrialFailure(userID);
  const cancellation = useTrialCancellation({ userID, query, failure });
  const { mutateAsync: checkout, isPending: isCheckoutPending } =
    usePostTrialsStartTrialCheckout();
  // The mutation settles before the redirect, while the conversion is still
  // going out; the button must stay busy until the page actually leaves.
  const [isCheckingOut, setIsCheckingOut] = useState(false);
  const isStarting = isCheckoutPending || isCheckingOut;
  const offer = query.data?.eligible ? query.data.offer : null;
  useTrialOfferViewed({ offer, userID, surface: returnTo });

  async function startTrial() {
    if (!offer || !userID || isStarting) return;
    failure.clearFailure();
    setIsCheckingOut(true);
    try {
      const response = await checkout({
        data: { offer_token: offer.token, return_to: returnTo },
      });
      if (useAuthStore.getState().user?.id !== userID) return;
      if (response.status !== 200) throw new Error(CHECKOUT_FAILED);
      markTrialCheckoutStarted(returnTo);
      await trackAdsConversionBeforeNavigation("begin_checkout", {
        value: getTrialChargeAmount(offer),
        currency: offer.currency.toUpperCase(),
      });
      window.location.assign(response.data.url);
    } catch (error) {
      failure.reportFailure({ userID, error, fallback: CHECKOUT_FAILED });
      await query.refetch();
    } finally {
      setIsCheckingOut(false);
    }
  }

  return {
    userID,
    trial: query.data,
    isLoading: Boolean(userID) && query.isLoading,
    error: failure.error,
    queryError: query.isError,
    retry: () => query.refetch(),
    isStarting,
    startTrial,
    ...cancellation,
  };
}
