import { usePlatformPayment } from "@/hooks/usePlatformPayment";
import { environment } from "@/services/environment";
import { useFlagResolutionStarted } from "@/services/feature-flags/flag-resolution";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef, useState } from "react";
import { buildStepLayout, type Step } from "./store";
import { usePaidCheckoutStatus } from "./usePaidCheckoutStatus";

// How long onboarding waits for the interview flag's answer, past the flag's
// own timeout (which serves its default, not an answer), before it lays the
// wizard out without the interview for this visit.
export const INTERVIEW_FLAG_WAIT_MS = 6_000;

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
  const hasWaitedForInterview = useHasWaited(
    INTERVIEW_FLAG_WAIT_MS,
    !interview.answered,
  );

  const flagsReady =
    payment.isResolved && (interview.answered || hasWaitedForInterview);
  const snapshot = useRef<{
    userID: string | null;
    payment: boolean;
    interview: boolean;
    interviewAnswered: boolean;
  } | null>(null);
  if (snapshot.current?.userID !== userID) snapshot.current = null;
  // Resolve once per account: a setting change must not reorder an active wizard.
  // A failed read is laid out as off but not frozen, so a later success still counts.
  if (!snapshot.current && flagsReady && !payment.isError && !isUserLoading) {
    snapshot.current = {
      userID,
      payment: payment.isPaymentEnabled,
      interview: interview.answered && interview.enabled,
      interviewAnswered: interview.answered,
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
    // The layout holds the interview flag's answer, not a wait that ran out.
    isLayoutAnswered: !!snapshot.current?.interviewAnswered,
  };
}

// Whether *ms* have passed, while *isWaiting*, since the flag service could
// start answering.
function useHasWaited(ms: number, isWaiting: boolean) {
  const isResolutionStarted = useFlagResolutionStarted();
  const [hasWaited, setHasWaited] = useState(false);
  useEffect(() => {
    if (!isResolutionStarted || !isWaiting) return;
    const timer = setTimeout(() => setHasWaited(true), ms);
    return () => clearTimeout(timer);
  }, [isResolutionStarted, isWaiting, ms]);
  return hasWaited;
}
