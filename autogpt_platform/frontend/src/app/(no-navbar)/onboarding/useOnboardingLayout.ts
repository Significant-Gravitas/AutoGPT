import { usePlatformPayment } from "@/hooks/usePlatformPayment";
import { environment } from "@/services/environment";
import { useFlagResolutionStarted } from "@/services/feature-flags/flag-resolution";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useEffect, useState } from "react";
import { useSignInProvider } from "./steps/AccessStep/useSignInProvider";
import { buildStepLayout, resolveStepOrder, type Step } from "./store";
import { useFrozenPerUser } from "./useFrozenPerUser";
import { usePaidCheckoutStatus } from "./usePaidCheckoutStatus";

// How long onboarding waits for the interview flag's answer, past the flag's
// own timeout (which serves its default, not an answer), before it lays the
// wizard out without the interview for this visit.
export const INTERVIEW_FLAG_WAIT_MS = 10_000;

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
  const stepOrder = useFlagStatus(Flag.ONBOARDING_STEP_ORDER);
  const signIn = useSignInProvider(userID);
  const interview = useFlagStatus(Flag.ONBOARDING_INTERVIEW);
  const hasWaitedForInterview = useHasWaited(
    INTERVIEW_FLAG_WAIT_MS,
    !interview.answered,
  );

  const flagsReady = payment.isResolved;
  // Resolved once per account: a setting or flag change must not reorder an
  // active wizard. Payment and placement resolve apart, so a checkout return
  // is confirmed without waiting on the sign-in lookup. A failed payment read
  // is laid out as off but not frozen, so a later success still counts.
  const hasPayment = useFrozenPerUser(
    userID,
    flagsReady && !payment.isError && !isUserLoading,
    () => payment.isPaymentEnabled,
  );
  const placement = useFrozenPerUser(
    userID,
    stepOrder.ready && signIn.isReady && !isUserLoading,
    () => ({
      requestedOrder: stepOrder.enabled,
      hasAccess: signIn.provider !== null,
    }),
  );
  // The interview flag's answer, or, once INTERVIEW_FLAG_WAIT_MS has run out
  // without one, no interview for this visit.
  const interviewLayout = useFrozenPerUser(
    userID,
    (interview.answered || hasWaitedForInterview) && !isUserLoading,
    () => ({
      hasInterview: interview.answered && interview.enabled,
      isAnswered: interview.answered,
    }),
  );
  // The paywall moves the access step, so the order follows payment as the
  // layout sees it: off until a read succeeds.
  const order = placement
    ? resolveStepOrder(placement.requestedOrder, { paywall: !!hasPayment })
    : undefined;
  const paidConfirmation = usePaidCheckoutStatus(
    userID,
    !!hasPayment && !environment.isLocal(),
  );
  const userHasActivePlan = trialConfirmation.active || paidConfirmation.active;
  const isPaymentEnabled = !!hasPayment && !userHasActivePlan;
  const isSelfHostConnectEnabled = !isPaymentEnabled && environment.isLocal();
  const steps = buildStepLayout({
    hasPaywall: isPaymentEnabled,
    // With no paywall configured the plan stands alone. A plan that is
    // already paid for was seen on the way to Stripe, so it is not repeated.
    hasPlan: !hasPayment,
    hasConnect: isSelfHostConnectEnabled,
    hasAccess: !!placement?.hasAccess,
    hasInterview: !!interviewLayout?.hasInterview,
    order,
  });
  return {
    steps,
    preparingStep: steps.preparing as Step,
    totalSteps: steps.preparing - 1,
    isReady:
      flagsReady &&
      !!placement &&
      !!interviewLayout &&
      !isUserLoading &&
      (!isLoggedIn || !paidConfirmation.isLoading) &&
      trialConfirmation.ready &&
      paidConfirmation.ready,
    paidConfirmation,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
    // The layout holds the interview flag's answer, not a wait that ran out.
    isLayoutAnswered: !!interviewLayout?.isAnswered,
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

