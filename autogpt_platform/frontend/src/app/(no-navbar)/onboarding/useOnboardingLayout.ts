import { environment } from "@/services/environment";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useRef } from "react";
import { usePaidCheckoutStatus } from "./usePaidCheckoutStatus";
import { buildStepLayout, type Step } from "./store";

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
  const brain = useFlagStatus(Flag.ONBOARDING_BRAIN_DUMP);
  const expertTeam = useFlagStatus(Flag.ONBOARDING_EXPERT_TEAM);
  const hireExperts = useFlagStatus(Flag.HIRE_EXPERTS);
  const flagsReady =
    payment.ready && brain.ready && expertTeam.ready && hireExperts.ready;
  const snapshot = useRef<{
    userID: string | null;
    payment: boolean;
    brain: boolean;
    team: boolean;
  } | null>(null);
  if (snapshot.current?.userID !== userID) snapshot.current = null;
  // Resolve once per account: flag changes must not reorder an active wizard.
  if (!snapshot.current && flagsReady && !isUserLoading) {
    snapshot.current = {
      userID,
      payment: payment.enabled,
      brain: brain.enabled,
      team: expertTeam.enabled && hireExperts.enabled,
    };
  }
  const paidConfirmation = usePaidCheckoutStatus(
    userID,
    !!snapshot.current?.payment && !environment.isLocal(),
  );
  const userHasActivePlan = trialConfirmation.active || paidConfirmation.active;
  const isPaymentEnabled = !!snapshot.current?.payment && !userHasActivePlan;
  const isBrainDumpEnabled = snapshot.current?.brain ?? false;
  const isExpertTeamEnabled = snapshot.current?.team ?? false;
  const isSelfHostConnectEnabled = !isPaymentEnabled && environment.isLocal();
  const steps = buildStepLayout({
    hasIntro: isExpertTeamEnabled,
    hasHire: isExpertTeamEnabled && isBrainDumpEnabled,
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
    isBrainDumpEnabled,
    isExpertTeamEnabled,
  };
}
