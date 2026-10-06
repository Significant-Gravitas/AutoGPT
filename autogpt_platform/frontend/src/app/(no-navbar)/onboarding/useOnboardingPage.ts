import {
  postV1CompleteOnboardingStep,
  postV1SubmitOnboardingProfile,
} from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackAdsConversion } from "@/services/analytics/google-ads";
import { trackTrialCheckoutAbandoned } from "@/services/analytics/monetization-analytics";
import { environment } from "@/services/environment";
import { useTrialCheckoutReturn } from "@/services/trials/useTrialCheckoutReturn";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { accountDisplayName, normalizeOnboardingProfile } from "./helpers";
import { Step, useOnboardingWizardStore } from "./store";
import { onboardingStepKey, trackOnboardingStep } from "./tracking";
import { stepKey } from "./progress";
import { useWizardProgress } from "./useWizardProgress";
import { useOnboardingLayout } from "./useOnboardingLayout";

export function useOnboardingPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const { isLoggedIn, isUserLoading, user } = useAuth();
  const trialConfirmation = useTrialCheckoutReturn();
  const storedStep = useOnboardingWizardStore((s) => s.currentStep);
  const goToStep = useOnboardingWizardStore((s) => s.goToStep);

  const {
    steps,
    preparingStep,
    totalSteps,
    isReady,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
    isBrainDumpEnabled,
    isExpertTeamEnabled,
    paidConfirmation,
  } = useOnboardingLayout({
    userID: user?.id ?? null,
    isLoggedIn,
    isUserLoading,
    trialConfirmation,
  });

  const progress = useWizardProgress({
    userID: user?.id ?? null,
    ready: isReady,
    steps,
  });
  const currentStep =
    isPaymentEnabled && !environment.isLocal()
      ? (Math.min(storedStep, steps.subscription!) as Step)
      : storedStep;
  const hasSubmitted = useRef<string | null>(null);
  const isCompleting = useRef(false);
  const activeUserID = useRef(user?.id);
  activeUserID.current = user?.id;
  const [completionError, setCompletionError] = useState<string | null>(null);

  useEffect(() => {
    if (!progress.isReady) return;
    if (currentStep !== storedStep) goToStep(currentStep);
    if (searchParams.get("trial") === "cancelled") {
      trackTrialCheckoutAbandoned("onboarding");
    }
    const key = stepKey(steps, currentStep);
    if (searchParams.get("step") !== key) {
      router.replace(`/onboarding?step=${key}`, { scroll: false });
    }
  }, [
    progress.isReady,
    currentStep,
    storedStep,
    goToStep,
    steps,
    searchParams,
    router,
  ]);

  const trackingKey = onboardingStepKey(steps, currentStep);
  useEffect(() => {
    if (progress.isReady && trackingKey) trackOnboardingStep(trackingKey);
  }, [progress.isReady, trackingKey]);

  // Submit profile when entering the Preparing step
  useEffect(() => {
    if (
      !progress.isReady ||
      currentStep !== preparingStep ||
      hasSubmitted.current === user?.id
    )
      return;
    const { role, painPoints } = normalizeOnboardingProfile(
      useOnboardingWizardStore.getState(),
    );
    const userName = accountDisplayName(user);

    // The profile is only ever submitted here, once, on reaching Preparing.
    // Guard against an empty role so a stray Preparing visit can't blank a
    // previously-saved profile.
    if (!role.trim() || !userName) return;
    hasSubmitted.current = user?.id ?? null;

    postV1SubmitOnboardingProfile({
      user_name: userName,
      user_role: role,
      pain_points: painPoints,
    }).catch(() => {
      // Best effort — profile data is non-critical for accessing copilot
    });
  }, [currentStep, preparingStep, user, progress.isReady]);

  async function handlePreparingComplete() {
    if (
      !progress.isReady ||
      isCompleting.current ||
      (isPaymentEnabled && !environment.isLocal())
    )
      return;
    isCompleting.current = true;
    setCompletionError(null);
    try {
      await useOnboardingWizardStore.getState().flushProgress?.();
      if (activeUserID.current !== user?.id) return;
      const result = await postV1CompleteOnboardingStep({
        step: "ONBOARDING_COMPLETE",
      });
      if (activeUserID.current !== user?.id) return;
      if (result.status !== 200) throw new Error("Completion failed");
      trackAdsConversion("onboarding_complete", {
        transactionID: user?.id,
        email: user?.email,
      });
      progress.finish();
      router.replace("/copilot");
    } catch {
      if (activeUserID.current !== user?.id) return;
      setCompletionError(
        "We couldn't finish setting up your account. Your progress is saved; please retry.",
      );
    } finally {
      isCompleting.current = false;
    }
  }

  return {
    currentStep,
    isLoading: !progress.isReady || !isReady,
    progressError: progress.error,
    progressConflict: progress.conflict,
    retryProgress: progress.retry,
    completionError,
    handlePreparingComplete,
    isPaymentEnabled,
    isSelfHostConnectEnabled,
    isBrainDumpEnabled,
    isExpertTeamEnabled,
    steps,
    preparingStep,
    totalSteps,
    trialConfirmation,
    paidConfirmation,
  };
}
