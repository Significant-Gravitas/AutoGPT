import { usePlatformPayment } from "@/hooks/usePlatformPayment";
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
  const payment = usePlatformPayment();
  const interview = useFlagStatus(Flag.ONBOARDING_INTERVIEW);

  const flagsReady = payment.isResolved && interview.ready;
  const snapshot = useRef<{
    userID: string | null;
    payment: boolean;
    interview: boolean;
  } | null>(null);
  if (snapshot.current?.userID !== userID) snapshot.current = null;
  // Resolve once per account: a setting change must not reorder an active wizard.
  // A failed read is laid out as off but not frozen, so a later success still counts.
  if (!snapshot.current && flagsReady && !payment.isError && !isUserLoading) {
    snapshot.current = {
      userID,
      payment: payment.isPaymentEnabled,
      interview: interview.enabled,
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
    // With no paywall configured the plan stands alone. A plan that is
    // already paid for was seen on the way to Stripe, so it is not repeated.
    hasPlan: !snapshot.current?.payment,
    hasConnect: isSelfHostConnectEnabled,
    hasInterview: !!snapshot.current?.interview,
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
