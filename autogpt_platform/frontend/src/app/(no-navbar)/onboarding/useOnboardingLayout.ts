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
  const payment = useFlagStatus(Flag.ENABLE_PLATFORM_PAYMENT);
  const stepOrder = useFlagStatus(Flag.ONBOARDING_STEP_ORDER);
  const signIn = useSignInProvider(isLoggedIn);

  const flagsReady = payment.ready;
  // Resolved once per account: flag changes must not reorder an active wizard.
  // Payment and placement resolve apart, so a checkout return is confirmed
  // without waiting on the sign-in lookup.
  const hasPayment = useFrozenPerUser(
    userID,
    flagsReady && !isUserLoading,
    () => payment.enabled,
  );
  const placement = useFrozenPerUser(
    userID,
    flagsReady && stepOrder.ready && signIn.isReady && !isUserLoading,
    () => ({
      order: resolveStepOrder(stepOrder.enabled, { paywall: payment.enabled }),
      hasAccess: signIn.provider !== null,
    }),
  );
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
