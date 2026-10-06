import { postV1CompleteOnboardingStep } from "@/app/api/__generated__/endpoints/onboarding/onboarding";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { trackAdsConversion } from "@/services/analytics/google-ads";
import { trackTrialCheckoutAbandoned } from "@/services/analytics/monetization-analytics";
import { environment } from "@/services/environment";
import { useTrialCheckoutReturn } from "@/services/trials/useTrialCheckoutReturn";
import { useRouter, useSearchParams } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { Step, useOnboardingWizardStore } from "./store";
import { onboardingStepKey, trackOnboardingStep } from "./tracking";
import { stepKey } from "./progress";
import { useWizardProgress } from "./useWizardProgress";
import { useOnboardingLayout } from "./useOnboardingLayout";
import { useOnboardingProfile } from "./useOnboardingProfile";

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
  const profile = useOnboardingProfile({
    user,
    enabled: progress.isReady && currentStep === preparingStep,
  });
  const completion = useRef<{ userID: string | undefined } | null>(null);
  const activeUserID = useRef(user?.id);
  activeUserID.current = user?.id;
  const [completionError, setCompletionError] = useState<{
    userID: string | undefined;
    message: string;
  } | null>(null);

  useEffect(() => {
    return () => {
      completion.current = null;
    };
  }, [user?.id]);

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

  async function handlePreparingComplete() {
    if (
      !progress.isReady ||
      completion.current?.userID === user?.id ||
      (isPaymentEnabled && !environment.isLocal())
    )
      return;
    const attempt = { userID: user?.id };
    completion.current = attempt;
    function isCurrent() {
      return (
        completion.current === attempt && activeUserID.current === user?.id
      );
    }
    setCompletionError(null);
    try {
      await useOnboardingWizardStore.getState().flushProgress?.();
      if (!isCurrent()) return;
      await profile.ensureSaved();
      if (!isCurrent()) return;
      const result = await postV1CompleteOnboardingStep({
        step: "ONBOARDING_COMPLETE",
      });
      if (!isCurrent()) return;
      if (result.status !== 200) throw new Error("Completion failed");
      trackAdsConversion("onboarding_complete", {
        transactionID: user?.id,
        email: user?.email,
      });
      progress.finish();
      router.replace("/copilot");
    } catch {
      if (!isCurrent()) return;
      setCompletionError({
        userID: user?.id,
        message:
          "We couldn't finish setting up your account. Your progress is saved; please retry.",
      });
    } finally {
      if (completion.current === attempt) completion.current = null;
    }
  }

  return {
    currentStep,
    isLoading: !progress.isReady || !isReady,
    progressError: progress.error,
    progressConflict: progress.conflict,
    retryProgress: progress.retry,
    completionError:
      profile.error ??
      (completionError?.userID === user?.id ? completionError?.message : null),
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
