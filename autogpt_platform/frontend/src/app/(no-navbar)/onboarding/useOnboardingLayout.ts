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
  // A failed read is laid out as off but not frozen, so a later success still
  // counts. The step placement follows the same rule, since the paywall moves
  // the access step.
  const paymentKnown = flagsReady && !payment.isError;
  const placementReady = stepOrder.ready && signIn.isReady && !isUserLoading;
  const readPlacement = () => ({
    order: resolveStepOrder(stepOrder.enabled, {
      paywall: payment.isPaymentEnabled,
    }),
    hasAccess: signIn.provider !== null,
  });
  // Resolved once per account: a setting or flag change must not reorder an
  // active wizard. Payment and placement resolve apart, so a checkout return
  // is confirmed without waiting on the sign-in lookup.
  const hasPayment = useFrozenPerUser(
    userID,
    paymentKnown && !isUserLoading,
    () => payment.isPaymentEnabled,
  );
  const frozenPlacement = useFrozenPerUser(
    userID,
    paymentKnown && placementReady,
    readPlacement,
  );
  const placement =
    frozenPlacement ?? (flagsReady && placementReady ? readPlacement() : null);
  const paidConfirmation = usePaidCheckoutStatus(
    userID,
    !!hasPayment && !environment.isLocal(),
  );
  const userHasActivePlan = trialConfirmation.active || paidConfirmation.active;
  const isPaymentEnabled = !!hasPayment && !userHasActivePlan;
  const isSelfHostConnectEnabled = !isPaymentEnabled && environment.isLocal();
  const steps = buildStepLayout({
    hasPaywall: isPaymentEnabled,
    hasConnect: isSelfHostConnectEnabled,
    hasAccess: !!placement?.hasAccess,
    order: placement?.order,
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
