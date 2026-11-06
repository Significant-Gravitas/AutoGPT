import { usePlatformPayment } from "@/hooks/usePlatformPayment";
import { environment } from "@/services/environment";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useSignInProvider } from "./steps/AccessStep/useSignInProvider";
import { buildStepLayout, resolveStepOrder, type Step } from "./store";
import { useFrozenPerUser } from "./useFrozenPerUser";
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
  const stepOrder = useFlagStatus(Flag.ONBOARDING_STEP_ORDER);
  const signIn = useSignInProvider(isLoggedIn);

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
    order,
  });
  return {
    steps,
    preparingStep: steps.preparing as Step,
    totalSteps: steps.preparing - 1,
    isReady:
      flagsReady &&
      !!placement &&
      !isUserLoading &&
      (!isLoggedIn || !paidConfirmation.isLoading) &&
      trialConfirmation.ready &&
      paidConfirmation.ready,
    paidConfirmation,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
  };
}
