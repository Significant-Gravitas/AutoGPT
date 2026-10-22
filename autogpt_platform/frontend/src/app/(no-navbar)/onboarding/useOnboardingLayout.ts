import { environment } from "@/services/environment";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useRef } from "react";
import { buildStepLayout, type Step } from "./store";
import { usePaidCheckoutStatus } from "./usePaidCheckoutStatus";

export function useOnboardingLayout({
  userID,
  isLoggedIn,
  isUserLoading,
  trialConfirmation,
}: {
  userID: string | null;
  isLoggedIn: boolean;
  isUserLoading: boolean;
  trialConfirmation: { ready: boolean; active: boolean | undefined };
}) {
  const payment = useFlagStatus(Flag.ENABLE_PLATFORM_PAYMENT);

  const flagsReady = payment.ready;
  const snapshot = useRef<{
    userID: string | null;
    payment: boolean;
  } | null>(null);
  if (snapshot.current?.userID !== userID) snapshot.current = null;
  // Resolve once per account: flag changes must not reorder an active wizard.
  if (!snapshot.current && flagsReady && !isUserLoading) {
    snapshot.current = {
      userID,
      payment: payment.enabled,
    };
  }
  const paidConfirmation = usePaidCheckoutStatus(
    userID,
    !!snapshot.current?.payment && !environment.isLocal(),
  );
  const userHasActivePlan = trialConfirmation.active || paidConfirmation.active;
  const isPaymentEnabled = !!snapshot.current?.payment && !userHasActivePlan;
  const isSelfHostConnectEnabled = !isPaymentEnabled && environment.isLocal();
  const steps = buildStepLayout({
    hasPaywall: isPaymentEnabled,
    hasConnect: isSelfHostConnectEnabled,
  });
  return {
    steps,
    preparingStep: steps.preparing as Step,
    totalSteps: steps.preparing - 1,
    isReady:
      flagsReady &&
      !isUserLoading &&
      (!isLoggedIn || !paidConfirmation.isLoading) &&
      trialConfirmation.ready &&
      paidConfirmation.ready,
    paidConfirmation,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
  };
}
